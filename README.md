# TSQRseqNovo:

**TSARseqNovo: A Transformer-Based Semi-Autoregressive Framework for High-Speed and Accurate De Novo Peptide Sequencing**
TSARseqNovo is a superior transformer-based method designed for high-speed and accurate de novo sequencing.

## Requirements
~~~
pytorch 1.9.0
pytorch-lightning 1.9.1
spectrum_utils 0.4.2 
depthcharge-ms 0.1.0
~~~
We recommend to use the model in Linux (ubuntu 22.04) with a gpu with cuda 11.8.

## Data
You can download the nine-species benchmark from the url: https://zenodo.org/records/6791263/files/casanovo_preprocessed_data.zip

## Usage
### Train
1. Prepare ~/TSARseqNovo/denovo/scripst/train/config.yaml, change train_data_path, valid_data_path, experiment_name and save_path. And change other training parameters as you wish.
2. run train script 
~~~
cd ~/TSARseqNovo
python train.py -c /train_path/config.yaml
~~~
### Inference
1. Prepare ~/TSARseqNovo/denovo/scripst/train/config.yaml, change resume(ckpt path of pretrained model), valid_data_path(try sample.hdf5 in the TSARseqNovo folder), experiment_name and save_path. And change other inference parameters as you wish. You can use sample.hdf5 as an example.
2. run predict script
~~~
cd ~/TSARseqNovo
python train.py -c /predict_path/config.yaml
~~~
