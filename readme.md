# FFCV Dataloader with Pytorch Lightning

FFCV is a fast dataloader for neural networks training: https://github.com/libffcv/ffcv  

In this repository, all the steps to install and configure it with pytorch-lightning are presented.  
The idea is to provide very generic methods and utils, while letting the user decide and configure anything.

## Installation

Tested with: 
```
Ubuntu 22.04.2 LTS
python 3.11
ffcv==1.0.2
pytorch==2.0.1
pytorch-lightning==2.0.4
```

### Dependencies

You can install dependencies (FFCV, Pytorch) with the provided `environment.yml` file:  
```
conda env create --file environment.yml
conda activate ffcv-pl
```
This should correctly create a conda environment named `ffcv-pl`.  

**Note:** Modify the pytorch-cuda version to the one compatible with your system.

**Note:** Solving environment can take quite a long time. 
I suggest to use [libmamba solver](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) 
to speed up the process.

**If the above does not work**, then another option is manual installation: 

1. create conda environment
    ```
    conda create --name ffcv-pl
    conda activate ffcv-pl
    ```

2. install pytorch according to [official website](https://pytorch.org/get-started/locally/) 

    ```
    # in my environment the command is the following 
    conda install pytorch torchvision torchaudio pytorch-cuda=[your-version] -c pytorch -c nvidia
    ```

3. install ffcv dependencies and pytorch-lightning
    ```
    # can take some time for solving, but should not create conflicts
    conda install cupy pkg-config libjpeg-turbo">=2.1.4" opencv numba pytorch-lightning">=2.0.0" -c pytorch -c conda-forge
    ```

4. install ffcv
    ```
    pip install ffcv
    ```

For further help, check out FFCV installation guidelines: [ffcv official page](https://github.com/libffcv/ffcv)

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

Example from the `dataset_creation.py` script:

```
from ffcv_pl.generate_dataset import create_beton_wrapper
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image


class ToyImageLabelDataset(Dataset):

    def __init__(self, n_samples: int):
        self.samples = [Image.fromarray((np.random.rand(32, 32, 3) * 255).astype('uint8')).convert('RGB')
                        for _ in range(n_samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (self.samples[idx], int(idx))


def main():
    
    # 1. Instantiate the torch dataset that you want to create
    # Important: the __get_item__ dataset must return tuples! (This depends on FFCV library)
    image_label_dataset = ToyImageLabelDataset(n_samples=256)

    # 2. call the method, and it will automatically create the .beton dataset for you.
    create_beton_wrapper(image_label_dataset, "./data/image_label.beton")

    
if __name__ == '__main__':
    
    main()
```

## Dataloader and Datamodule

Merge the PL Datamodule with the FFCV Loader object.  
Official FFCV Loader [docs](https://docs.ffcv.io/making_dataloaders.html).   
Official Pytorch-Lightning DataModule [docs](https://lightning.ai/docs/pytorch/stable/data/datamodule.html).

In `main.py` a complete example on how to use the `FFCVDataModule` method and train a 
Lightning Model is given.

The main steps to follow are:
1. create `FFCVPipelineManager` object, which needs the path to a previously created `.beton` file, 
   a list of operations to perform on each item returned by your dataset and an ordering option for Loading.
2. create the `FFCVDataModule` object, which is a Lightning Module with FFCV Loader.
3. Pass the data module to Pytorch Lightning trainer, and run!

**Suggestion** : read FFCV [performance guide](https://docs.ffcv.io/performance_guide.html) to better
   understand which options fit your needs.

Complete Example from the `main.py` script:

```
import pytorch_lightning as pl
import torch
from ffcv.fields.basics import IntDecoder
from ffcv.fields.rgb_image import RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
from ffcv.loader import OrderOption
from ffcv.transforms import ToTensor, ToTorchImage
from pytorch_lightning.strategies.ddp import DDPStrategy

from torch import nn
from torch.optim import Adam
from torchvision.transforms import RandomHorizontalFlip

from ffcv_pl.data_loading import FFCVDataModule
from ffcv_pl.ffcv_utils.augmentations import DivideImage255

from ffcv_pl.ffcv_utils.utils import FFCVPipelineManager


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(32 * 32 * 3, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 32 * 32 * 3))

    def training_step(self, batch, batch_idx):

        x = batch[0]

        b, c, h, w = x.shape
        x = x.reshape(b, -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)

        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():

    seed = 1234

    pl.seed_everything(seed, workers=True)

    batch_size = 16
    gpus = 2
    nodes = 1
    workers = 8

    # image label dataset
    train_manager = FFCVPipelineManager("./data/image_label.beton",  # previously defined using dataset_creation.py
                                        pipeline_transforms=[

                                            # image pipeline
                                            [RandomResizedCropRGBImageDecoder((32, 32)),
                                             ToTensor(),
                                             ToTorchImage(),
                                             DivideImage255(dtype=torch.float32),
                                             RandomHorizontalFlip(p=0.5)],

                                            # label (int) pipeline
                                            [IntDecoder(),
                                             ToTensor()
                                             ]
                                        ],
                                        ordering=OrderOption.RANDOM)  # random ordering for training

    val_manager = FFCVPipelineManager("./data/image_label.beton",
                                      pipeline_transforms=[

                                          # image pipeline (different from train)
                                          [CenterCropRGBImageDecoder((32, 32), ratio=1.),
                                           ToTensor(),
                                           ToTorchImage(),
                                           DivideImage255(dtype=torch.float32)],

                                          # label (int) pipeline
                                          None  # if None, uses default
                                      ],
                                      ordering=OrderOption.SEQUENTIAL)  # sequential ordering for validation

    # datamodule creation
    # ignore test and predict steps, since managers are not defined.
    data_module = FFCVDataModule(batch_size, workers, train_manager=train_manager, val_manager=val_manager,
                                 is_dist=True, seed=seed)

    # define model
    model = LitAutoEncoder()

    # trainer
    trainer = pl.Trainer(strategy=DDPStrategy(find_unused_parameters=False), deterministic=True,
                         accelerator='gpu', devices=gpus, num_nodes=nodes, max_epochs=5, logger=False)

    # start training!
    trainer.fit(model, data_module)


if __name__ == '__main__':

    main()

```

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