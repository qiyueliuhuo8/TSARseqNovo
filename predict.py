import argparse
import yaml
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from denovo.model.model_runner import predict_MSDataset

def run_predict(cfg):
    valid_data_path = cfg["dataset"]["valid_data_path"]
   
    predict_MSDataset(valid_data_path, cfg)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train RSAM")
    parser.add_argument('--cfg_path', '-c', default='./denovo/scripts/predict/config.yaml', metavar='CFG_PATH',
                        type=str,
                        help='Path to config file in yaml format')
    args = parser.parse_args()
    cfg = yaml.load(open(args.cfg_path, 'r'), Loader=yaml.Loader)
    
    run_predict(cfg)