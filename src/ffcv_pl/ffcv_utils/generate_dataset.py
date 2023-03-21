import numpy as np

from ffcv import DatasetWriter
from ffcv.fields import RGBImageField, JSONField, NDArrayField

from ffcv_pl.datasets.image import ImageDataset
from ffcv_pl.datasets.image_label import ImageLabelDataset
from ffcv_pl.datasets.image_array import ImageArrayDataset


def create_image_dataset(train_folder: str = None, validation_folder: str = None, test_folder: str = None):
    """
    create train/validation and optionally test .beton file to the father directory of each path.
    E.G.:
        train_folder = '/home/datasets/imagenet/train/' --> '/home/datasets/imagenet/train.beton'
    """

    folders = [train_folder, validation_folder, test_folder]
    names = ['train.beton', 'validation.beton', 'test.beton']

    for f, n in zip(folders, names):

        if f is not None:
            path = f"{'/'.join(f.split('/')[:-2])}/{n}"
            dataset = ImageDataset(f)

            # official guidelines: https://docs.ffcv.io/writing_datasets.html
            print(f'creating dataset from folder: {f}\tto file: {path}')
            writer = DatasetWriter(path, {'image': RGBImageField(write_mode='jpg')})
            writer.from_indexed_dataset(dataset)
        else:
            print(f'No folder name provided for {n}')


def create_image_label_dataset(train_folder: str = None, validation_folder: str = None, test_folder: str = None):
    """
    create train/validation and optionally test .beton file to the father directory of each path.
    E.G.:
        train_folder = '/home/datasets/imagenet/train/' --> '/home/datasets/imagenet/train.beton'

    folder must be organized as follows (to correctly get class name):
            /home/datasets/imagenet/train/
                                    class1/
                                            images
                                    class2/
                                            images
                                    classn/
                                            images
    """

    folders = [train_folder, validation_folder, test_folder]
    names = ['train.beton', 'validation.beton', 'test.beton']

    for f, n in zip(folders, names):

        if f is not None:
            path = f"{'/'.join(f.split('/')[:-2])}/{n}"
            dataset = ImageLabelDataset(f)

            # official guidelines: https://docs.ffcv.io/writing_datasets.html
            print(f'creating dataset from folder: {f}\tto file: {path}')
            writer = DatasetWriter(path, {'image': RGBImageField(write_mode='jpg'), 'label': JSONField()})
            writer.from_indexed_dataset(dataset)
        else:
            print(f'No folder name provided for {n}')


def create_image_array_dataset(array_shape: tuple, array_dtype: np.dtype = np.dtype('float32'),
                               train_folder: str = None, validation_folder: str = None, test_folder: str = None):
    """
    create train/validation and optionally test .beton file to the father directory of each path.
    E.G.:
        train_folder = '/home/datasets/imagenet/train/' --> '/home/datasets/imagenet/train.beton'

    folder must be organized as follows (to correctly get image/numpy array couples):
            /home/datasets/my_custom_dataset/train/
                                                /0001.npy
                                                /0001.png
                                                /0002.npy
                                                /0002.png
                                                ...
                                                /wxyz.npy
                                                /wxyz.png
    :param array_shape: shape of numpy arrays (all must have the same shape)
    :param array_dtype: dtype of numpy arrays (all must have the same dtype - default to np.dtype('float32'))
    """

    folders = [train_folder, validation_folder, test_folder]
    names = ['train.beton', 'validation.beton', 'test.beton']

    for f, n in zip(folders, names):

        if f is not None:
            path = f"{'/'.join(f.split('/')[:-2])}/{n}"
            dataset = ImageArrayDataset(f)

            # official guidelines: https://docs.ffcv.io/writing_datasets.html
            print(f'creating dataset from folder: {f}\tto file: {path}')
            writer = DatasetWriter(path, {'image': RGBImageField(write_mode='jpg'),
                                          'array': NDArrayField(array_dtype, array_shape)})
            writer.from_indexed_dataset(dataset)
        else:
            print(f'No folder name provided for {n}')
