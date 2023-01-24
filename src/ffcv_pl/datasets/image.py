import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from glob import glob
from PIL import Image
import numpy as np

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToTorchImage
from ffcv.fields.decoders import CenterCropRGBImageDecoder

from ffcv_pl.ffcv_utils.custom_augmentations import DivideImage255


class ImageDataset(Dataset):

    def __init__(self, folder: str):
        """
        :param folder: path to images in [.png, .jpg, .jPEG] formats
        """

        self.image_names = glob(f'{folder}*.jpg', recursive=True) + glob(f'{folder}*.png', recursive=True) + \
                           glob(f'{folder}*.JPEG', recursive=True)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        """
        :param index: the image index to return
        :return: the corresponding image and optionally class name
        """

        # load image with no preprocessing.
        # return type should be a Tuple
        image = (np.array(Image.open(self.image_names[index]).convert('RGB')), )

        return image


class ImageDataModule(pl.LightningDataModule):

    def __init__(self, train_file: str, val_file: str, test_file: str, image_size: int, dtype: torch.dtype,
                 batch_size: int, num_workers: int, is_dist: bool, seed: int):
        """
        :param train_file: path to .beton train file
        :param val_file: path to .beton validation file
        :param test_file: path to .beton test file
        :param image_size: all loaded images will be returned with this size
        :param dtype: torch desired type for returned images. May be one of torch.float16, torch.float32, torch.float64
        """

        super().__init__()

        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file

        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.pipeline = None
        self.dtype = dtype

        self.is_dist = is_dist
        self.seed = seed

    def setup(self, stage=None):

        # prepare data
        image_pipeline = [CenterCropRGBImageDecoder((self.image_size, self.image_size), ratio=1.),
                          ToTensor(), ToTorchImage(), DivideImage255(self.dtype)]

        # Pipeline for each data field
        self.pipeline = {
            'image': image_pipeline,
        }

    def train_dataloader(self):
        return Loader(self.train_file, batch_size=self.batch_size, num_workers=self.num_workers,
                      order=OrderOption.RANDOM, pipelines=self.pipeline, distributed=self.is_dist, seed=self.seed)

    def val_dataloader(self):
        return Loader(self.val_file, batch_size=self.batch_size, num_workers=self.num_workers,
                      order=OrderOption.SEQUENTIAL, pipelines=self.pipeline, distributed=self.is_dist, seed=self.seed)

    def test_dataloader(self):
        return Loader(self.test_file, batch_size=self.batch_size, num_workers=self.num_workers,
                      order=OrderOption.SEQUENTIAL, pipelines=self.pipeline, distributed=self.is_dist, seed=self.seed)
