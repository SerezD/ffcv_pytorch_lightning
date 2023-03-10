# FFCV Dataloader with Pytorch Lightning

FFCV is a fast dataloader for neural networks training: https://github.com/libffcv/ffcv  

In this repository, all the steps to install and configure it with pytorch-lightning are presented.  
Moreover, some useful methods to quickly create, preprocess and load Datasets with *FFCV* and *pytorch-lightning* 
are proposed.

## Installation

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

**If the above does not work**, then 
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

3. install ffcv dependencies and pytorch-lightning
    ```
    # can take a very long time, but should not create conflicts
    conda install cupy pkg-config compilers libjpeg-turbo opencv numba pytorch-lightning -c pytorch -c conda-forge
    ```

4. install ffcv
    ```
    pip install ffcv
    ```

### Package

Once dependencies are installed, it is safe to install package: 
```
pip install ffcv_pl
```

## Dataset Creation

You need to save your dataset in ffcv format (`.beton`).  
Official FFCV [docs](https://docs.ffcv.io/writing_datasets.html).

This package allows different types of Datasets, listed in the `dataset` subpackage.
A quick example on how to create a dataset is provided in the `dataset_creation.py script`:

```
from ffcv_pl.ffcv_utils.generate_dataset import create_image_label_dataset

if __name__ == '__main__':

    # write dataset in ".beton" format
    train_folder = '/media/dserez/datasets/cub/train/'
    test_folder = '/media/dserez/datasets/cub/test/'
    create_image_label_dataset(train_folder=train_folder, test_folder=test_folder)
```

For example, this code will create the files `/media/dserez/datasets/cub/test.beton` and 
`/media/dserez/datasets/cub/train.beton`, 
loading images from folders `/media/dserez/datasets/cub/test/` and 
`/media/dserez/datasets/cub/train/`, respectively.

Note that you can pass also more folders, all in one call. 

## Dataloader and Datamodule

Merge the PL Datamodule with the FFCV Loader object.  
It should be compatible with ddp/multiprocessing.  
See `main.py` for a complete example.  
Official FFCV [docs](https://docs.ffcv.io/making_dataloaders.html).

```
import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies.ddp import DDPStrategy

from torch import nn
from torch.optim import Adam

from ffcv_pl.datasets.image import ImageDataModule


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):

    def __init__(self):

        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(256 * 256 * 3, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 256 * 256 * 3))

    def training_step(self, batch, batch_idx):

        x, y = batch

        b, c, h, w = x.shape
        x = x.reshape(b, -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':

    SEED = 1234

    pl.seed_everything(SEED, workers=True)

    dataset = 'cub'
    image_size = 256
    batch_size = 16
    train_folder = f'/media/dserez/datasets/{dataset}/train.beton'
    val_folder = f'/media/dserez/datasets/{dataset}/test.beton'

    gpus = 2
    workers = 8

    # define model
    model = LitAutoEncoder()

    # trainer
    trainer = pl.Trainer(strategy=DDPStrategy(find_unused_parameters=False), deterministic=True,
                         accelerator='gpu', devices=gpus, num_nodes=1, max_epochs=5)

    # Note: set is_dist True if you are using DDP and more than one GPU
    data_module = ImageDataModule(train_folder, val_folder, val_folder, image_size, torch.float32, batch_size,
                                  num_workers=1, is_dist=gpus > 1, seed=SEED)

    trainer.fit(model, data_module)

```

Each `ffcv_pl.datasets.*` contains a couple of classes (Dataset, Dataloader).

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