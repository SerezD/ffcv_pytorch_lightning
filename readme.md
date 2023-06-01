# FFCV Dataloader with Pytorch Lightning

FFCV is a fast dataloader for neural networks training: https://github.com/libffcv/ffcv  

In this repository, all the steps to install and configure it with pytorch-lightning are presented.  
The idea is to provide very generic methods and utils, while letting the user decide and configure anything.

## Installation

Tested with: 
```
Ubuntu 22.04.2 LTS
ffcv==1.0.2
pytorch==2.0.1
pytorch-lightning==2.0.2
```

### Dependencies

There are actually some known issues about the installation of the FFCV package.   
In particular, even a successful installation may rise the following error when 
trying to import `ffcv` (this seems to happen also in version `1.0.x` of FFCV):

```ImportError: libopencv_imgproc.so.405: cannot open shared object file: No such file or directory```

There is a Closed issue about this [#136](https://github.com/libffcv/ffcv/issues/136).

In order to correctly install everything, I suggest to use Conda 
(I tried also pip but encountered the error above).

First, try to install dependencies with `environment.yml` file:  
```
conda env create --file environment.yml
```
This should correctly create a conda environment named `ffcv-pl`.  

**Note:** Solving environment can take quite a long time. I suggest to use [libmamba solver](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) 
to speed up the process.

**If the above does not work**, then 
you can try installing packages manually (works with python 3.10): 

1. create conda environment
    ```
    conda create --name ffcv-pl python=3.10
    conda activate ffcv-pl
    ```

2. install pytorch according to [official website](https://pytorch.org/get-started/locally/) 

    ```
    # in my environment the command is the following 
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

3. install ffcv dependencies and pytorch-lightning
    ```
    # can take some time for solving, but should not create conflicts
    conda install cupy pkg-config libjpeg-turbo opencv numba pytorch-lightning">=2.0.0" -c pytorch -c conda-forge
    ```

4. install ffcv
    ```
    pip install ffcv
    ```

### Package

Once dependencies are installed, it is safe to install the package: 
```
pip install ffcv_pl
```

## Dataset Creation

You need to save your dataset in ffcv format (`.beton`).  
Official FFCV [docs](https://docs.ffcv.io/writing_datasets.html).

This package provides you the `create_beton_wrapper` method, which allows to easily create
a `.beton` dataset from a `torch` dataset.     
An example is given in the `dataset_creation.py` script.

## Dataloader and Datamodule

Merge the PL Datamodule with the FFCV Loader object.  
Official FFCV Loader [docs](https://docs.ffcv.io/making_dataloaders.html).   
Official Pytorch-Lightning DataModule [docs](https://lightning.ai/docs/pytorch/stable/data/datamodule.html).

In `main.py` a complete example on how to use the `FFCVDataModule` method and train a 
Lightning Model is given.

The main steps to follow are:
1. get the `.beton` files for the Loaders that you need (train, val, test or predict). 
   The file must be created with the `create_beton_wrapper` method.
2. get the corresponding `ffcv.Fields` types (same specified in the dataset creation method)
3. Optionally create the `FFCVDecoders` objects that defines the pipeline to apply.   
   You are free to select different transforms for train/val/test/predict.   
   See the Official FFCV Loader docs for more information.
4. call the `FFCVDataModule` method specifying the `.beton` files, the `FFCVDecoders`, the 
   `ffcv.Fields` and any other option of the FFCV Loader.  
   Also, read FFCV [performance guide](https://docs.ffcv.io/performance_guide.html) to better
   understand which options fit your needs.
5. Pass the data module to Pytorch Lightning, as you normally would!

## Code Citations

1. Pytorch-Lightning:
    ```
   @software{Falcon_PyTorch_Lightning_2019,
    author = {Falcon, William and {The PyTorch Lightning team}},
    doi = {10.5281/zenodo.3828935},
    license = {Apache-2.0},
    month = mar,
    title = {{PyTorch Lightning}},
    url = {https://github.com/Lightning-AI/lightning},
    version = {1.4},
    year = {2019}
    }
   ```

2. FFCV: 
    ```
    @misc{leclerc2022ffcv,
        author = {Guillaume Leclerc and Andrew Ilyas and Logan Engstrom and Sung Min Park and Hadi Salman and Aleksander Madry},
        title = {{FFCV}: Accelerating Training by Removing Data Bottlenecks},
        year = {2022},
        howpublished = {\url{https://github.com/libffcv/ffcv/}},
        note = {commit 2544abdcc9ce77db12fecfcf9135496c648a7cd5}
    }
    ```