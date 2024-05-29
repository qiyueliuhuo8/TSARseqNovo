from collections import OrderedDict
from typing import Any, Optional, Tuple, Union, List
import re
import math
import einops
import logging
import sys

import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Transformer
import pytorch_lightning as pl
import einops
from depthcharge.components.encoders import FloatEncoder, PeakEncoder, PositionalEncoder

logger = logging.getLogger("RSAM")

class T_Net(nn.Module):
    def __init__(self, k=8, dim=256):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(k, dim, 1)
        self.conv2 = torch.nn.Conv1d(dim, dim*2, 1)
        self.conv3 = torch.nn.Conv1d(dim*2, dim*4, 1)
        self.fc1 = nn.Linear(dim*4, dim*2)
        self.fc2 = nn.Linear(dim*2, dim)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim*2)
        self.bn3 = nn.BatchNorm1d(dim*4)
        self.bn4 = nn.BatchNorm1d(dim*2)
        self.bn5 = nn.BatchNorm1d(dim)

        self.k = k

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.permute(0,2,1)

        x = F.relu(self.bn4(self.fc1(x).permute(0,2,1)))
        x = x.permute(0,2,1)
        x = F.relu(self.bn5(self.fc2(x).permute(0,2,1)))
        return x.permute(0,2,1)
    
    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if name.endswith("bn5.weight"):
                        nn.init.zeros_(param)

class RSAM(nn.Module):
    def __init__(
        self, 
        dim_model,
        n_layers,
        n_heads,
        dim_aa_embedding,
        batch_first,
        dropout,
        max_out_len,
        batch_size,
        max_charge,
        k_step,
        train_label_smoothing
        ) -> None:
        """
        dim_model: int = 768,
        n_layers: int = 6,
        n_heads: int = 8,
        batch_first: bool = True,
        dropout: float = 0.1,
        max_out_len: int = 70,
        batch_size: int = 128,
        """
        super().__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_heads,
            dim_feedforward=4 * dim_model,
            dropout=dropout,
            batch_first=batch_first
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=n_layers
        )

        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_heads,
            dim_feedforward=4 * dim_model,
            dropout=dropout,
            batch_first=batch_first
        )
        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layers,
        )
        
        self.mr_decoder = torch.nn.TransformerDecoder(
            decoder_layer=decoder_layer, 
            num_layers=n_layers
        )

        # self.tnet = torch.nn.Linear(6, dim_model-dim_aa_embedding)
        self.tnet = T_Net(k=6,dim=dim_model-dim_aa_embedding)
        self.tnet.initialize_parameters()
        self.mass_encoder = FloatEncoder(dim_model-dim_aa_embedding) 
        self.precursor_mass_encoder = FloatEncoder(dim_model) 
        self.pos_encoder = PositionalEncoder(dim_model)
        

        self.vocabulary = {
            "G": 57.021464,
            "A": 71.037114,
            "S": 87.032028,
            "P": 97.052764,
            "V": 99.068414,
            "T": 101.047670,
            "C+57.021": 160.030649, # 103.009185 + 57.021464
            "L": 113.084064,
            "I": 113.084064,
            "N": 114.042927,
            "D": 115.026943,
            "Q": 128.058578,
            "K": 128.094963,
            "E": 129.042593,
            "M": 131.040485,
            "H": 137.058912,
            "F": 147.068414,
            "R": 156.101111,
            "Y": 163.063329,
            "W": 186.079313,
            # Amino acid modifications.
            "M+15.995": 147.035400,    # Met oxidation:   131.040485 + 15.994915
            "N+0.984": 115.026943,     # Asn deamidation: 114.042927 +  0.984016
            "Q+0.984": 129.042594,     # Gln deamidation: 128.058578 +  0.984016
            # N-terminal modifications.
            "+42.011": 42.010565,      # Acetylation
            "+43.006": 43.005814,      # Carbamylation
            "-17.027": -17.026549,     # NH3 loss
            "+43.006-17.027": 25.980265,      # Carbamylation and NH3 loss
        }

        self.max_length = max_out_len
        self._amino_acids = list(self.vocabulary.keys()) + ["$"]
        self._idx2aa = {i + 1: aa for i, aa in enumerate(self._amino_acids)} # 0 for annotations padding
        self._aa2idx = {aa: i for i, aa in self._idx2aa.items()}
        self.stop_token = self._aa2idx["$"]
        self.k_step = k_step
        self.bs = batch_size
        self.transformer_width = dim_model
        self.mass_embedding_dim = dim_model-dim_aa_embedding

        self.fc = torch.nn.Linear(dim_model, len(self._amino_acids) + 1)
        self.charge_encoder = torch.nn.Embedding(max_charge, dim_model)
        self.aa_encoder = torch.nn.Embedding(
            len(self._amino_acids) + 1,
            dim_aa_embedding,
            padding_idx=0,
        )
        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, dim_model))
        self.mask_embedding = torch.nn.Parameter(torch.randn(dim_aa_embedding))
        self.ln = nn.LayerNorm(dim_model)

        self.set_buffer(min_wavelength=0.001, max_wavelength=10000, buffer_type="mz", dim=dim_model-dim_aa_embedding)
        self.set_buffer(min_wavelength=1e-6, max_wavelength=1, buffer_type="inten", dim=dim_model-dim_aa_embedding)

        self.softmax = nn.Softmax(dim=2)

        self.celoss = torch.nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=train_label_smoothing
        )
        self.val_celoss = torch.nn.CrossEntropyLoss(ignore_index=0)

    @property
    def device(self):
        """The current device for the model"""
        return next(self.parameters()).device
    
    def initialize_parameters(self):
        nn.init.normal_(self.fc.weight, std=self.transformer_width ** -0.5)
        # nn.init.normal_(self.tnet.weight, std=self.transformer_width ** -0.5)

        proj_std = (self.transformer_width ** -0.5) * ((2 * self.n_layers) ** -0.5)
        attn_std = self.transformer_width ** -0.5
        fc_std = (2 * self.transformer_width) ** -0.5
        for block in self.encoder.layers:
            nn.init.normal_(block.self_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.self_attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.linear1.weight, std=fc_std)
            nn.init.normal_(block.linear2.weight, std=proj_std)

        for block in self.decoder.layers:
            nn.init.normal_(block.self_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.self_attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.multihead_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.multihead_attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.linear1.weight, std=fc_std)
            nn.init.normal_(block.linear2.weight, std=proj_std)

        for block in self.mr_decoder.layers:
            nn.init.normal_(block.self_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.self_attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.multihead_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.multihead_attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.linear1.weight, std=fc_std)
            nn.init.normal_(block.linear2.weight, std=proj_std)


    def tokenize(self, sequence, partial=False):
        if not isinstance(sequence, str):
            return sequence  # Assume it is already tokenized.

        sequence = sequence.replace("I", "L")
        sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)
        for i in range(self.k_step):
            sequence +=["$"]

        tokens = [self._aa2idx[aa] for aa in sequence]
        tokens = torch.tensor(tokens, device=self.device)
        return tokens

    def detokenize(self, tokens):
        sequence = [self._idx2aa.get(i.item(), "") for i in tokens]

        return sequence
    
    def turn_mass(self, sequence):
        if isinstance(sequence, str):
            sequence = sequence.replace("I", "L")
            sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)
            for i in range(self.k_step):
                sequence +=["$"]

        masses = []
        for s in sequence:
            if s == "$":
                masses.append(float(0))
            else:
                masses.append(self.vocabulary[s])
        masses = torch.tensor(masses, device=self.device)
        return masses
    
    def turn_idx_mass(self, tokens):
        masses = torch.full((tokens.shape), 0)
        for i in range(masses.shape[0]):
            for j in range(masses.shape[1]):
                aa = self._idx2aa[tokens[i,j].item()]
                if aa == '$':
                    masses[i,j] = 0
                else:
                    masses[i,j] = self.vocabulary[aa]
        return masses.to(tokens.device)
        
    
    def set_buffer(self, min_wavelength, max_wavelength, buffer_type, dim):
        dim_model = dim
        d_sin = math.ceil(dim_model / 2)
        d_cos = math.ceil(dim_model / 2)

        base = min_wavelength / (2 * np.pi)
        scale = max_wavelength / min_wavelength
        sin_exp = torch.arange(0, d_sin).float() / (d_sin - 1)
        cos_exp = (torch.arange(d_sin, dim_model).float() - d_sin) / (
            d_cos - 1
        )
        sin_term = base * (scale**sin_exp)
        cos_term = base * (scale**cos_exp)

        self.register_buffer(buffer_type + "_sin_term", sin_term)
        self.register_buffer(buffer_type + "_cos_term", cos_term)

    def encode_ms(self, ms):
        m_over_z = ms[:, :, 0]
        sin_mz = torch.sin(m_over_z[:, :, None] / self.mz_sin_term)
        cos_mz = torch.cos(m_over_z[:, :, None] / self.mz_cos_term)
        encoded_mz = torch.cat([sin_mz, cos_mz], axis=-1)
        
        int_input = ms[:, :, 1]
        int_sin_mz = torch.sin(int_input[:, :, None] / self.inten_sin_term)
        int_cos_mz = torch.cos(int_input[:, :, None] / self.inten_cos_term)
        encoded_int = torch.cat([int_sin_mz, int_cos_mz], axis=-1)
        # x = encoded_mz + encoded_int
        x = torch.cat([encoded_mz, encoded_int], dim=-1)
        return x
    
    def encoder_forward(self, ms, features):
        
        zeros = ~ms.sum(dim=2).bool()
        mask = [
            torch.tensor([[False]] * ms.shape[0]).type_as(zeros),
            zeros,
        ]
        mask = torch.cat(mask, dim=1)
        encoded_ms = self.encode_ms(ms)
        latent_spectra = self.latent_spectrum.expand(encoded_ms.shape[0], -1, -1)

        feature_embeddings = self.tnet(features.permute(0,2,1))
        inputs = torch.cat([encoded_ms, feature_embeddings], dim=-1)
        inputs = torch.cat([latent_spectra, inputs], dim=1)
        inputs = self.ln(inputs)

        inputs = inputs.permute(1,0,2)
        return self.encoder(inputs, src_key_padding_mask=mask), mask
    
    def decoder_forward(self, sequences, precursors, memory, memory_key_padding_mask, tgt_padd_msk = None):
         # Prepare mass and charge
        masses = self.precursor_mass_encoder(precursors[:, None, 0])
        charges = self.charge_encoder(precursors[:, 1].int() - 1)
        precursors = masses + charges[:, None, :]
        precursors = precursors.repeat(1,self.k_step,1)

        if sequences is None:
            tgt = precursors
        else:
            if isinstance(sequences, torch.Tensor):
                tokens = sequences
                aa_masses = self.turn_idx_mass(tokens)
            else:
                aa_masses = [self.turn_mass(s) for s in sequences]
                aa_masses = torch.nn.utils.rnn.pad_sequence(aa_masses, batch_first=True)
                tokens = [self.tokenize(s) for s in sequences]
                tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True)
            tgt = torch.cat([self.aa_encoder(tokens), self.mass_encoder(aa_masses)], dim=-1)
            tgt = torch.cat([precursors, tgt], dim=1)

        if sequences is None: 
            tgt_key_padding_mask = torch.zeros((tgt.shape[0], tgt.shape[1])).bool().to(self.device)
            if tgt_padd_msk is not None:
                tgt_key_padding_mask = tgt_padd_msk    
        else:
            tgt_key_padding_mask = ~tokens.bool()
            tgt_key_padding_mask = torch.cat(
                [torch.zeros((aa_masses.shape[0], self.k_step)).bool().to(self.device), tgt_key_padding_mask],
                dim=1
            )
        

        tgt = self.pos_encoder(tgt)
        tgt_mask = generate_tgt_mask(tgt.shape[1], self.k_step).to(self.device)

        preds = self.decoder(
            tgt=tgt.permute(1,0,2),
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask.to(self.device),
        )

        if sequences is None:
            return self.fc(preds).permute(1,0,2), tgt_key_padding_mask
        return self.fc(preds).permute(1,0,2), tokens, tgt_key_padding_mask

    def mr_decoder_forward(self,decoder_out, memory, tgt_key_padding_mask, memory_key_padding_mask, real_tokens, mask_ratio=0.1):
        decoder_out = decoder_out[:,:-(self.k_step),1:]
        decoder_out = self.softmax(decoder_out)
        tmp = torch.topk(decoder_out, 1, dim=-1)
        decoder_tokens = tmp[1].squeeze(-1) + 1

        tmp = tmp[0].squeeze(-1)
        true_matrix = torch.ones(tmp.shape).bool().to(decoder_out.device)
        false_matrix = torch.zeros(tmp.shape).bool().to(decoder_out.device)
        false_mask = torch.where(tmp<mask_ratio, true_matrix, false_matrix).squeeze(-1)

        zero_dim = int(self.mass_embedding_dim/2)
        one_dim = self.mass_embedding_dim-zero_dim
        zero_mass_embedding = torch.concat([torch.zeros(zero_dim), torch.ones(one_dim)]).to(decoder_out.device)
        mask_embedidng = torch.concat([self.mask_embedding,zero_mass_embedding])

        aa_masses = []
        for i in range(decoder_out.shape[0]):
            aa_masses.append(self.detokenize(decoder_tokens[i]))
        aa_masses = [self.turn_mass(seq) for seq in aa_masses]
        aa_masses = torch.cat(aa_masses, dim=0).reshape(decoder_out.shape[0],-1)
        tgt = torch.cat([self.aa_encoder(decoder_tokens), self.mass_encoder(aa_masses)], dim=-1)

        tgt[false_mask] = mask_embedidng

        refine_preds = self.mr_decoder(
            tgt = tgt.permute(1,0,2),
            memory = memory,
            tgt_key_padding_mask = tgt_key_padding_mask,
            memory_key_padding_mask = memory_key_padding_mask.to(self.device),
        )
        return self.fc(refine_preds).permute(1,0,2), decoder_tokens

    def forward(self, ms, precursors, annotations, features, mask_ratio):
        memories, memory_mask = self.encoder_forward(ms, features)
        decoder_out, tokens, tgt_padding_mask = self.decoder_forward(annotations, precursors, memories, memory_mask)
        refine_out, decoder_tokens = self.mr_decoder_forward(decoder_out, memories, tgt_padding_mask[:,(self.k_step):], memory_mask, tokens, mask_ratio)
        decoder_out = decoder_out[:, : -(self.k_step), :].reshape(-1, len(self._amino_acids) + 1)
        mr_tokens = torch.topk(refine_out[:,:,1:], 1, dim=-1)[1].squeeze(-1) + 1

        refine_out = refine_out[:, : , :].reshape(-1, len(self._amino_acids) + 1)
        loss_decoder = self.celoss(decoder_out, tokens.flatten())
        loss_mr = self.celoss(refine_out, tokens.flatten())
        loss = loss_decoder + loss_mr
        
        return loss, mr_tokens

def generate_tgt_mask(length, k):
    m = torch.ones(length, length, dtype=torch.bool)
    kk = (length + k - 1) // k
    for i in range(kk):
        if i == kk - 1:
            m[i * k: , :] = False
        else:
            m[i * k: i * k + k, :i * k + k] = False
    return m

