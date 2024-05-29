import argparse
import yaml
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from denovo.model.model_runner import train_MSDataset
def run_train(cfg):
    train_data_path = cfg["dataset"]["train_data_path"]
    valid_data_path = cfg["dataset"]["valid_data_path"]
    model_name = cfg["model"]["model_name"]
    train_MSDataset(train_data_path, valid_data_path, model_name, cfg)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train RSAM")
    parser.add_argument('--cfg_path', '-c', default='./denovo/scripts/train/config.yaml', metavar='CFG_PATH',
                        type=str,
                        help='Path to config file in yaml format')
    args = parser.parse_args()
    cfg = yaml.load(open(args.cfg_path, 'r'), Loader=yaml.Loader)
    
    run_train(cfg)