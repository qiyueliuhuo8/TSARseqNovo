# TSQRseqNovo:

**TSARseqNovo: A Transformer-Based Semi-Autoregressive Framework for High-Speed and Accurate De No**
TSARseqNovo is a superior transformer-based method designed for high-speed and accurate de novo sequencing.


## Usage
### Train
1. Prepare ~/TSARseqNovo/denovo/scripst/train/config.yaml, change train_data_path, valid_data_path, experiment_name and save_path. And change other training parameters as you wish.
2. run train script 
~~~
cd ~/TSARseqNovo
python train.py -c /train_path/config.yaml
~~~
### Inference
1. Prepare ~/TSARseqNovo/denovo/scripst/train/config.yaml, change resume(ckpt path of pretrained model), valid_data_path(try sample.hdf5 in the TSARseqNovo folder), experiment_name and save_path. And change other inference parameters as you wish.
2. run predict script
~~~
cd ~/TSARseqNovo
python train.py -c /predict_path/config.yaml
~~~