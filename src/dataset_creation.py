from ffcv.fields import RGBImageField

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

    # 2. Optional: create Field objects.
    # here overwrites only RGBImageField, leave default IntField.
    fields = (RGBImageField(write_mode='jpg', max_resolution=32), None)

    # 3. call the method, and it will automatically create the .beton dataset for you.
    create_beton_wrapper(image_label_dataset, "./data/image_label.beton", fields)


if __name__ == '__main__':

    main()

