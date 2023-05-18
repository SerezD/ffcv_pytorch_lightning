from ffcv.fields import RGBImageField, IntField, NDArrayField, TorchTensorField, JSONField
from ffcv_pl.generate_dataset import create_beton_wrapper
from torch.utils.data.dataset import Dataset
import numpy as np
import torch
from PIL import Image


class ToyImageLabelDataset(Dataset):

    def __init__(self, n_samples: int):
        self.samples = [Image.fromarray((np.random.rand(32, 32, 3) * 255).astype('uint8')).convert('RGB')
                        for _ in range(n_samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (self.samples[idx], int(idx))


class ToyArrayDataset(Dataset):

    def __init__(self, n_samples: int):
        self.samples = np.random.rand(n_samples, 32, 32, 3).astype('float32')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (self.samples[idx],)


class ToyTensorJsonDataset(Dataset):

    def __init__(self, n_samples: int):
        self.samples = torch.rand((n_samples, 32, 32, 3), dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (self.samples[idx], {'str_label': str(idx)})


if __name__ == '__main__':

    # 1. Instantiate the torch dataset that you want to create (here are three toy examples)
    # Important: the __get_item__ dataset must return tuples! (This depends on FFCV library)
    image_label_dataset = ToyImageLabelDataset(n_samples=256)
    array_dataset = ToyArrayDataset(n_samples=256)
    tensor_json_dataset = ToyTensorJsonDataset(n_samples=256)

    # 2. call the method, and it will automatically create the .beton dataset for you.
    # for each item that the __get_item__ method returns, you must provide a Field object
    # https://docs.ffcv.io/api/fields.html
    create_beton_wrapper(image_label_dataset, "./image_label",
                         (RGBImageField(write_mode='jpg'),
                          IntField()))

    create_beton_wrapper(array_dataset, "./array.beton",
                         (NDArrayField(shape=(32, 32, 3), dtype=np.dtype('float32')),))

    create_beton_wrapper(tensor_json_dataset, "./tensor_bytes",
                         (TorchTensorField(shape=(32, 32, 3), dtype=torch.float32),
                          JSONField()))
