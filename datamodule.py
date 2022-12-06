import pytorch_lightning as pl

from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToTorchImage
from ffcv.fields.decoders import CenterCropRGBImageDecoder

from custom_augmentations import DivideImage255


class ImageDataModule(pl.LightningDataModule):

    def __init__(self, train_folder: str, test_folder: str, image_size: int, batch_size: int, num_workers: int,
                 is_dist: bool, seed: int):
        """
        :param train_folder: path to train.beton
        :param test_folder: path to test.beton
        """

        super().__init__()

        self.train_folder = train_folder
        self.test_folder = test_folder

        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.pipeline = None

        self.is_dist = is_dist
        self.seed = seed

    def setup(self, stage=None):

        # prepare data
        # Data decoding and augmentation
        # Note: created custom augmentation to set images in range 0__1 and not 0__255
        image_pipeline = [CenterCropRGBImageDecoder((self.image_size, self.image_size), ratio=1.),
                          ToTensor(), ToTorchImage(), DivideImage255()]

        # Pipeline for each data field
        self.pipeline = {
            'image': image_pipeline,
        }

    def train_dataloader(self):
        return Loader(self.train_folder, batch_size=self.batch_size, num_workers=self.num_workers,
                      order=OrderOption.RANDOM, pipelines=self.pipeline, distributed=self.is_dist, seed=self.seed)

    def val_dataloader(self):
        return Loader(self.test_folder, batch_size=self.batch_size, num_workers=self.num_workers,
                      order=OrderOption.SEQUENTIAL, pipelines=self.pipeline, distributed=self.is_dist, seed=self.seed)

