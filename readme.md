# FFCV Dataloader with Pytorch Lightning

FFCV is a fast dataloader for neural networks training: https://github.com/libffcv/ffcv  

In this repository, all the steps to install and configure it with pytorch-lightning are presented.  
Versions:
- FFCV 0.0.3
- Pytorch Lightning 1.8 

## Package installation

A first problem may arise with packages installations, since some conflicts may appear. 
Check for instance issues of FFCV ([#133](https://github.com/libffcv/ffcv/issues/133) 
[#54](https://github.com/libffcv/ffcv/issues/54)). 

```
# create conda environment
conda create --name ffcv-pl
conda activate ffcv-pl

# install pytorch according to official website (https://pytorch.org/get-started/locally/)
# in my environment the command is the following 
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

# install ffcv dependencies (can take a very long time, but should not create conflicts)
conda install cupy pkg-config compilers libjpeg-turbo opencv numba -c pytorch -c conda-forge

# install ffcv
pip install ffcv

# install pytorch-lightning and other packages (if needed)
pip install pytorch-lightning
```

## Dataset Creation

You need to save your dataset in ffcv format (`.beton`).   
A quick example is presented in `dataset_creation.py`  
Official FFCV [docs](https://docs.ffcv.io/writing_datasets.html).

## Dataloader and Datamodule

Merge the PL Datamodule with the FFCV Loader object.  
It should be compatible with ddp/multiprocessing.  
See `datamodule.py` for a complete example.  
Official FFCV [docs](https://docs.ffcv.io/making_dataloaders.html).

## Launch Training

See `main.py` for a dummy example.  
Basically just a standard PL train script, everything has been set up in datamodule.  
