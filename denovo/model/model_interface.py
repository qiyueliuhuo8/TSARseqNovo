import heapq
import inspect
import einops
import torch
import importlib
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import warnings
import math
import collections
import sys

from torch.nn import functional as F
import torch.optim.lr_scheduler as lrs
import numpy as np
import pytorch_lightning as pl

from .. import evaluate

class MInterface(pl.LightningModule):
    def __init__(self, model_name, lr, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()

        self.mask_ratio = 0.1
        self.max_out_len = self.model.max_length
        self.stop_token = self.model.stop_token
        self.k = self.model.k_step
        self.softmax = torch.nn.Softmax(2)

        self.test_predictions = []

    def training_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor, List[str], torch.Tensor],
        *args,
        mode: str = "train",
    ) -> torch.Tensor:
        ms, precursors, annotations, features = batch
        loss,_ = self.model.forward(ms, precursors, annotations, features, self.mask_ratio)
          
        self.log(
            name="train_loss",
            value=loss.detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )

        lr = self.optimizers().optimizer.param_groups[0]['lr']
        self.log(name="lr", value=lr, on_step=True, sync_dist=True)
        
        return loss
    
    def validation_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor, List[str], torch.Tensor], *args,mode: str = "valid",
    ) -> torch.Tensor:
        ms, precursors, annotations, features = batch
        loss, mr_tokens = self.model.forward(ms, precursors, annotations, features, self.mask_ratio)
        
        pred_peptides = []
        for i in range(mr_tokens.shape[0]):
            pred_peptides.append("".join(self.detokenize(mr_tokens[i])))
        aa_precision, _, pep_precision = evaluate.aa_match_metrics(
            *evaluate.aa_match_batch(
                pred_peptides,
                annotations,
                self.model.vocabulary,
            )
        )
        self.log(
            name="aa_precision",
            value=aa_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        self.log(
            name="pep_precision",
            value=pep_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        self.log(
            name="valid_loss",
            value=loss.detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True
        )
        return loss
    
    def predict_step(
            self, batch: Tuple[torch.Tensor, torch.Tensor, List[str], torch.Tensor], *args,mode: str = "predict",
    ) -> torch.Tensor:
        ms, precursors, annotations, features = batch
        mr_tokens = self.decode_stop_early_noreal(ms, precursors, features)

        pred_peptides = []
        for i in range(mr_tokens.shape[0]):
            pred_peptides.append("".join(self.detokenize(mr_tokens[i])))
        aa_precision, _, pep_precision = evaluate.aa_match_metrics(
            *evaluate.aa_match_batch(
                pred_peptides,
                annotations,
                self.model.vocabulary,
            )
        )
        aa_matches_batch, _, _ = evaluate.aa_match_batch(
                pred_peptides,
                annotations,
                self.model.vocabulary,
            )
        matches = [aa_matches[1] for aa_matches in aa_matches_batch]

        print(f"aa precision: {aa_precision}")
        print(f"pep precision: {pep_precision}")

        self.test_predictions.append((pred_peptides, annotations, matches))
        return (pred_peptides, annotations, matches)
    
    def decode_stop_early_noreal(self, ms: torch.Tensor, precursors: torch.Tensor, features: torch.Tensor):
        memories, memory_mask = self.model.encoder_forward(ms, features)
        batch = ms.shape[0]
        vocab = len(self.model._aa2idx) + 1
        length = self.max_out_len
        k = self.model.k_step
        steps = length // k

        pred, _ = self.model.decoder_forward(None, precursors, memories, memory_mask)
        tokens = torch.zeros((batch, length), dtype=torch.int64).to(pred.device)
        tokens[:,:k,] = torch.topk(pred[:, :, 1:], 1, dim=-1)[1].squeeze(-1) + 1
        scores = torch.full(size=(batch, length, vocab), fill_value=torch.nan).to(pred.device)
        scores[:,:k,:] = pred

        total_finsh = torch.zeros(batch, dtype=torch.bool).to(pred.device)
        for step in range(0, steps-1):
            begin_step = step*self.k
            stop_step = (step+1) * k  

            tmp = (tokens[:, begin_step:stop_step] == self.stop_token)
            ends_stop_token = tmp.sum(-1).bool()
            finished_beams = torch.zeros(tokens.shape[0], dtype=torch.bool).to(
                tokens.device
            )
            finished_beams[ends_stop_token] = True
            total_finsh |= finished_beams
            if total_finsh.all():
                break

            scores[~total_finsh, :stop_step+k, :], _, _,= self.model.decoder_forward(
                tokens[~total_finsh, :stop_step],
                precursors[~total_finsh, :],
                memories[:, ~total_finsh, :],
                memory_mask[~total_finsh, :],
        )
            tokens[~total_finsh, stop_step:stop_step+k] = torch.topk(scores[~total_finsh, stop_step:stop_step+k, 1:], 1, dim=-1)[1].squeeze(-1) + 1
        tgt_key_padding_mask = ~tokens.bool()
        tt = tgt_key_padding_mask.sum(1).max().detach().cpu().numpy().tolist()
        refine_out, decoder_tokens = self.model.mr_decoder_forward(scores[:,:tt+2], memories, tgt_key_padding_mask[:,:tt], memory_mask, None, mask_ratio=self.mask_ratio)

        mr_tokens = torch.topk(refine_out[:,:,1:], 1, dim=-1)[1].squeeze(-1) + 1
        return mr_tokens

    def get_tgt_padding_msk(self, sequences, k ):
        tokens = [self.model.tokenize(s) for s in sequences]
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
        if tokens.shape[1] % k != 0:
            over_k = k - (tokens.shape[1] % k)
            tokens = torch.cat([tokens, torch.zeros((tokens.shape[0],over_k), dtype=torch.int64).to(tokens.device)], dim=1)

        tgt_key_padding_mask = ~tokens.bool()
        tgt_key_padding_mask = torch.cat(
                [torch.zeros((tokens.shape[0], self.k)).bool().to(tokens.device), tgt_key_padding_mask],
                dim=1
            )
        return tgt_key_padding_mask, tokens

    def on_validation_epoch_end(self) -> None:
        metrics = self.trainer.callback_metrics
        aa_precsion = metrics["aa_precision"].detach().item()
        if aa_precsion > 0.8:
            self.mask_ratio = 0.8
        elif aa_precsion < 0.1:
            self.mask_ratio = 0.1
        else:
            self.mask_ratio = int(aa_precsion*100)/100
        self.log(name="mask_ratio",
            value=self.mask_ratio,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True)

    def detokenize(self, tokens):
        sequence = [self.model._idx2aa.get(i.item(), "") for i in tokens]
        if "$" in sequence:
            idx = sequence.index("$")
            sequence = sequence[: idx]

        return sequence

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.lr_decay_steps,
                                       gamma=self.hparams.lr_decay_rate)
            elif self.hparams.lr_scheduler == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.lr_decay_steps,
                                                  eta_min=self.hparams.lr_decay_min_lr)
            elif self.hparams.lr_scheduler == 'cosine_custom':
                scheduler = CosineWarmupScheduler(
                    optimizer, warmup=self.hparams.warmup_iters, max_iters=self.hparams.max_iters
                )
            elif self.hparams.lr_scheduler == 'warmup_cosine_restart':
                scheduler = WarmupCosineAnnealingWarmRestarts(
                    optimizer, T_0=self.hparams.max_iters, warmup_iter=self.hparams.warmup_iters, 
                    T_mult=self.hparams.t_mult, eta_min=self.hparams.lr_min
                )
            elif self.hparams.lr_scheduler == 'warmup':
                scheduler = WarmupScheduler(
                    optimizer, warmup=self.hparams.warmup_iters, max_iters=self.hparams.max_iters,
                )
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], {"scheduler": scheduler, "interval": "step"}

    def load_model(self):
        model_name = self.hparams.model_name
        try:
            Model = getattr(importlib.import_module(
                '.model', package=__package__), model_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name model.{model_name}!')
        self.model = self.instancialize(Model)

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getfullargspec(Model.__init__).args[1:]
        # class_args = inspect.signature(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm up followed by cosine shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup : int
        The number of warm up iterations.
    max_iters : torch.optim
        The total number of iterations.
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, warmup: int, max_iters: int
    ):
        self.warmup, self.max_iters = warmup, max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(iter=self._step_count)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, iter):
        lr_factor = 1
        if iter <= self.warmup:
            lr_factor *= iter / self.warmup
        return lr_factor

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm up followed by cosine shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup : int
        The number of warm up iterations.
    max_iters : torch.optim
        The total number of iterations.
    """

    def __init__(
        self, optimizer: torch.optim.Optimizer, warmup: int, max_iters: int
    ):
        self.warmup, self.max_iters = warmup, max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(iter=self._step_count)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, iter):
        lr_factor = 1e-5 + 0.5 * (1 + np.cos(np.pi * iter / self.max_iters))
        if iter <= self.warmup:
            lr_factor *= iter / self.warmup
        return lr_factor
    
class WarmupCosineAnnealingWarmRestarts(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, T_0, warmup_iter, T_mult=1, eta_min=0, last_epoch=-1, verbose=False):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        self.warmup_iter = warmup_iter
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]

    def step(self, epoch=None):

        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        if epoch < self.warmup_iter:
            if epoch == 0:
                lr = min(self.eta_min, (1 / self.warmup_iter) * self.base_lrs[0])
            else :
                lr = (epoch / self.warmup_iter) * self.base_lrs[0]

            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)
        else:
            with _enable_get_lr_call(self):
                for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                    param_group, lr = data
                    param_group['lr'] = lr
                    self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

