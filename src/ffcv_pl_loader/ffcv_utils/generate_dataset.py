from ffcv import DatasetWriter
from ffcv.fields import RGBImageField

from src.ffcv_pl_loader.datasets.image import ImageDataset


def create_image_dataset(train_folder: str, validation_folder: str = None, test_folder: str = None):
    """
    create train/validation and optionally test .beton file to the father directory of each path.
    E.G.:
        train_folder = 'home/datasets/imagenet/train/' --> 'home/datasets/imagenet/train.beton'
    """

    folders = [train_folder, validation_folder, test_folder]
    names = ['train.beton', 'validation.beton', 'test.beton']

    for f, n in zip(folders, names):

        if f is not None:
            path = '/'.join(f.split('/')[:-1]) + n
            dataset = ImageDataset(f)

            # official guidelines: https://docs.ffcv.io/writing_datasets.html
            writer = DatasetWriter(path, {'image': RGBImageField(write_mode='jpg')})
            writer.from_indexed_dataset(dataset)


# def write_to_path(path: str, dataset: Dataset):
#
#     writer = DatasetWriter(path, {'image': RGBImageField(write_mode='jpg', max_resolution=256, jpeg_quality=100),
#                                   'class': JSONField()})
#     writer.from_indexed_dataset(dataset)
