# FFCV Dataloader with Pytorch Lightning

FFCV is a fast dataloader for neural networks training: https://github.com/libffcv/ffcv  

In this repository, all the steps to install and configure it with pytorch-lightning are presented.  
Moreover, some useful methods to quickly create, preprocess and load Datasets with *FFCV* and *pytorch-lightning* 
are proposed.

## Package installation

There are actually some known issues about the installation of the FFCV package.   
Check for instance issues of FFCV ([#133](https://github.com/libffcv/ffcv/issues/133) 
[#54](https://github.com/libffcv/ffcv/issues/54)). 

The first suggestion to install dependencies for this repository is to use the provided `environment.yml` file:  
```
conda env create --file environment.yml
```
This should correctly create a conda environment named `ffcv-pl`. If the above does not work, then 
you can try installing packages manually: 

1. create conda environment
    ```
    conda create --name ffcv-pl
    conda activate ffcv-pl
    ```

2. install pytorch according to [official website](https://pytorch.org/get-started/locally/)

    ```
    # in my environment the command is the following 
    conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
    ```

3. install ffcv dependencies 
    ```
    # can take a very long time, but should not create conflicts
    conda install cupy pkg-config compilers libjpeg-turbo opencv numba -c pytorch -c conda-forge
    ```

4. install ffcv and pytorch-lighting
    ```
    pip install ffcv
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


## Citations

1. Pytorch-Lightning:  
    Falcon, W., & The PyTorch Lightning team. (2019). PyTorch Lightning (Version 1.4) 
    [Computer software]. https://doi.org/10.5281/zenodo.3828935

2. FFCV: 
    ```
    @misc{leclerc2022ffcv,
        author = {Guillaume Leclerc and Andrew Ilyas and Logan Engstrom and Sung Min Park and Hadi Salman and Aleksander Madry},
        title = {{FFCV}: Accelerating Training by Removing Data Bottlenecks},
        year = {2022},
        howpublished = {\url{https://github.com/libffcv/ffcv/}},
        note = {commit xxxxxxx}
    }
    ```