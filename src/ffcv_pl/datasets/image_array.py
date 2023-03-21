import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pathlib import Path
from PIL import Image
import numpy as np

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToTorchImage
from ffcv.fields.decoders import CenterCropRGBImageDecoder, NDArrayDecoder

from ffcv_pl.ffcv_utils.custom_augmentations import DivideImage255


class ImageArrayDataset(Dataset):

    def __init__(self, folder: str):
        """
        create a dataset to return (image, np_array) pairs. May be used for example to associate GAN generations and
        the corresponding latent.

        Assumes that the folders contains couples of image files and .npy files, with the same name

        E.G.
        /home/dataset/gan_generations/
                                /0001.npy
                                /0001.png
                                /0002.npy
                                /0002.png
                                ...
                                /wxyz.npy
                                /wxyz.png

        :param folder: path to images in [.png, .jpg, .jPEG] formats and [.npy] array files.
        """
        self.image_names = [str(p.resolve()) for p in Path(folder).rglob('*.JPEG')] + \
                                  [str(p.resolve()) for p in Path(folder).rglob('*.jpg')] + \
                                  [str(p.resolve()) for p in Path(folder).rglob('*.png')]

        # check correctness
        npy_names = [str(p.resolve()) for p in Path(folder).rglob('*.npy')]

        assert len(self.image_names) == len(npy_names), \
            f'In "{folder}" there are {len(self.image_names)} images and {len(npy_names)} numpy arrays.\n' \
            f'Please ensure that each image has only one corresponding array and vice-versa.'

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        """
        :param index: the image index to return
        :return: the corresponding (image, npy) couple
        """

        # load image
        image_tensor = np.array(Image.open(self.image_names[index]).convert('RGB'))

        # load array
        np_array = np.load('.'.join(self.image_names[index].split('.')[:-1]) + '.npy')

        return image_tensor, np_array


class ImageArrayDataModule(pl.LightningDataModule):

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
        self.seed = seed
        self.is_dist = is_dist
        self.dtype = dtype

        self.pipeline = None

    def setup(self, stage=None):
        # prepare data
        img_pipeline = [CenterCropRGBImageDecoder(output_size=(self.image_size, self.image_size), ratio=1.),
                        ToTensor(),
                        ToTorchImage(),
                        DivideImage255(self.dtype),
                        ]

        array_pipeline = [NDArrayDecoder()]

        self.pipeline = {
            'image': img_pipeline,
            'array': array_pipeline
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
