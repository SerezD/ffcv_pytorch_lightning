from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, Cutout, NormalizeImage
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder


from torch.utils.data import Dataset
from glob import glob
from PIL import Image
import numpy as np


def write_to_path(path: str, dataset: Dataset, image_resolution: int):

    # official guidelines: https://docs.ffcv.io/writing_datasets.html
    # can use a WEBDATASET instead, can save images as raw and not jpeg etc...
    writer = DatasetWriter(path, {'image': RGBImageField(write_mode='jpg', max_resolution=image_resolution)})
    writer.from_indexed_dataset(dataset)


class ImageDataset(Dataset):

    def __init__(self, folder: str):
        """
        :param folder: path to images
        """

        self.image_names = glob(folder + '*.png') + glob(folder + '*.jpg') + \
                           glob(folder + '*.bmp') + glob(folder + '*.JPEG')

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        """
        :param index: the image index to return
        :return: the corresponding image and optionally class name
        """

        # load image
        image_tensor = np.array(Image.open(self.image_names[index]).convert('RGB'))

        # IMPORTANT NOTE:
        # it seems like the return type must be a tuple.
        # moreover, here images are numpy.arrays with type np.uint8, normalized in 0__255 range
        # later preprocess will take care of conversion/normalization etc..
        return (image_tensor, )


if __name__ == '__main__':

    # write dataset in ".beton" format
    # this is a dummy example -- customize as you prefer
    train_folder = '/path/to/train/folder/images/'
    test_folder = '/path/to/test/folder/images/'

    final_train_dataset = './train.beton'
    final_test_dataset = './test.beton'

    # these operation can take several minutes, depending on dataset size
    write_to_path(final_train_dataset, dataset=ImageDataset(folder=train_folder), image_resolution=256)
    write_to_path(final_test_dataset, dataset=ImageDataset(folder=test_folder), image_resolution=256)
