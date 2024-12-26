# TSQRseqNovo:

**TSARseqNovo: A Transformer-Based Semi-Autoregressive Framework for High-Speed and Accurate De Novo Peptide Sequencing**

TSARseqNovo is a superior transformer-based method designed for high-speed and accurate de novo sequencing.

## System requirements
For operating systems, we recommend Ubuntu 20.04.6 LTS.
For envrionment prepare, please use docker (Docker 27.0.3). First, you need to create a clean container for docker. Then, use following command to install packages
~~~
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
~~~
For hardware, we recommend a GPU RTX 3090.

## Installation Guide
1. First, you need to clone the subject to your computer, use the following command:
~~~
git clone https://github.com/qiyueliuhuo8/TSARseqNovo.git
~~~
2. Download the nine-species benchmark from the url.
~~~
cd TSARseqNovo
mkdir data
cd data
wget https://zenodo.org/records/6791263/files/casanovo_preprocessed_data.zip
unzip casanovo_preprocessed_data.zip
~~~
These steps will take you a few hours depending on your internet speed.

## Demo
Here are the instructions for how to predict peptides with a trained model on the sample data.
1. modify the config.yaml under /path/to/TSARseqNovo/denovo/scripst/predict, the config.yaml is like
~~~
model:
  # replace with your own model
  resume: "../RSAM_ckpt/test.ckpt"

dataset:
  # replace with the true path of sample data
  valid_data_path: ["./TSARseqNovo/sample.hdf5"]
  max_out_len: 100

hparameters:
  batch_size: 128

dataloader:
  n_workers: 15

Trainer:
  device: '1'

experiment_name: 'predict'
# replace with your own log_path
save_path: './TSARseqNovo/logs'
~~~
2. run predict script
~~~
python predict.py -c /path/to/TSARseqNovo/denovo/scripst/predict/config.yaml
~~~
3. The expected output on the shell is like this, and the predicted peptides are saved as psms.csv under the log path that you set.
~~~
aa recall :
aa precision:
peptide precision: 
~~~
The predict process will take you a few minuntes depending on your gpu. If you use RTX 3090, it should take you less than 1 minute.

## Instructions for use
If you want to train the model on your own dataset, following the instructions below.
### Train
1. Prepare /path/to/TSARseqNovo/denovo/scripst/train/config.yaml, change train_data_path, valid_data_path, experiment_name and save_path. And change other training parameters as you wish.
2. run train script 
~~~
python train.py -c /train_path/config.yaml
~~~
The model checkpoints are saved under the log path, you can choose the best model for inference.
### Inference
1. Prepare ~/TSARseqNovo/denovo/scripst/train/config.yaml, change resume(ckpt path of pretrained model), valid_data_path(try sample.hdf5 in the TSARseqNovo folder), experiment_name and save_path. And change other inference parameters as you wish. You can use sample.hdf5 as an example.
2. run predict script
~~~
python predict.py -c /path/to/TSARseqNovo/denovo/scripst/predict/config.yaml
~~~
The peptide predictions are saved under the log path.
