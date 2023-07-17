import os
import shutil

from ffcv.fields.basics import IntDecoder, IntField
from ffcv.fields.bytes import BytesDecoder
from ffcv.fields.ndarray import NDArrayDecoder
from ffcv.loader import OrderOption

from ffcv_pl.ffcv_utils.utils import FFCVPipelineManager
from ffcv_pl.generate_dataset import create_beton_wrapper
from torch.utils.data.dataset import Dataset
import numpy as np
import torch
from PIL import Image
import pytorch_lightning as pl
from ffcv.fields.rgb_image import RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder, RGBImageField
from ffcv.transforms import ToTensor, ToTorchImage
from pytorch_lightning.strategies.ddp import DDPStrategy

from torch import nn
from torch.optim import Adam
from torchvision.transforms import RandomHorizontalFlip

from ffcv_pl.data_loading import FFCVDataModule
from ffcv_pl.ffcv_utils.augmentations import DivideImage255


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


class ToyIntStrDataset(Dataset):

    def __init__(self, n_samples: int):
        self.samples = [f'{i}' for i in range(n_samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return (idx, self.samples[idx])


class NotTupleDataset(Dataset):

    def __init__(self, n_samples: int):
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return idx


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


if __name__ == '__main__':

    # 1. Instantiate the torch dataset that you want to create (here are some toy examples)
    # Important: the __get_item__ dataset must return tuples! (This depends on FFCV library)
    image_label_dataset = ToyImageLabelDataset(n_samples=256)
    array_dataset = ToyArrayDataset(n_samples=256)
    tensor_json_dataset = ToyTensorJsonDataset(n_samples=256)
    int_str_dataset = ToyIntStrDataset(n_samples=256)
    bad_defined_dataset = NotTupleDataset(n_samples=256)

    # 2. call the method, and it will automatically create the .beton dataset for you.
    create_beton_wrapper(image_label_dataset, "./data/image_label.beton")

    create_beton_wrapper(array_dataset, "./data/array.beton")

    create_beton_wrapper(tensor_json_dataset, "./data/tensor_bytes.beton")

    try:
        create_beton_wrapper(int_str_dataset, "./data/int_str.beton")
    except AttributeError as e:
        print(f'Test passed: {e}')

    try:
        create_beton_wrapper(bad_defined_dataset, "./data/bad_defined.beton")
    except AttributeError as e:
        print(f'Test passed: {e}')

    # 2.1 check fields optional argument
    image_field = RGBImageField(write_mode='jpg', max_resolution=32)
    int_field = IntField()

    try:
        create_beton_wrapper(image_label_dataset, "./data/image_label.beton", fields=(image_field, ))
    except AttributeError as e:
        print(f'Test passed: {e}')

    create_beton_wrapper(image_label_dataset, "./data/image_label.beton", fields=(image_field, None))
    create_beton_wrapper(image_label_dataset, "./data/image_label.beton", fields=(image_field, int_field))

    # 3. define params
    SEED = 1234

    pl.seed_everything(SEED, workers=True)

    batch_size = 16
    gpus = 2
    nodes = 1
    workers = 8

    # 4. define all datamodules and test

    # no manager passed
    try:
        data_module = FFCVDataModule(batch_size, workers, is_dist=True, seed=SEED)
    except AttributeError as e:
        print(f'Test passed: {e}')

    # wrong number of pipelines passed
    try:
        manager = FFCVPipelineManager("./data/image_label.beton", pipeline_transforms=[None, ])
    except AttributeError as e:
        print(f'Test passed: {e}')

    # image label dataset
    train_manager = FFCVPipelineManager("./data/image_label.beton",
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
                                        ordering=OrderOption.RANDOM)

    val_manager = FFCVPipelineManager("./data/image_label.beton",
                                      pipeline_transforms=[

                                          # image pipeline
                                          [CenterCropRGBImageDecoder((32, 32), ratio=1.),
                                           ToTensor(),
                                           ToTorchImage(),
                                           DivideImage255(dtype=torch.float32)],

                                          # label (int) pipeline
                                          None
                                      ])

    # try Loader kwargs
    data_module = FFCVDataModule(batch_size, workers, train_manager=train_manager, val_manager=val_manager,
                                 is_dist=True, seed=SEED, train_drop_last=False, val_batches_ahead=5,
                                 not_exists_test=True)

    # define model
    model = LitAutoEncoder()

    # trainer
    trainer = pl.Trainer(strategy=DDPStrategy(find_unused_parameters=False), deterministic=True,
                         accelerator='gpu', devices=gpus, num_nodes=nodes, max_epochs=1, logger=False)

    trainer.fit(model, data_module)

    # array dataset case.
    train_manager = FFCVPipelineManager("./data/array.beton",
                                        pipeline_transforms=[
                                            # array pipeline
                                            [NDArrayDecoder(), ToTensor()],
                                        ],
                                        ordering=OrderOption.RANDOM)

    val_manager = FFCVPipelineManager("./data/array.beton", pipeline_transforms=[None, ])

    data_module = FFCVDataModule(batch_size, workers, train_manager=train_manager, val_manager=val_manager,
                                 is_dist=True, seed=SEED)

    # define model
    model = LitAutoEncoder()

    # trainer
    trainer = pl.Trainer(strategy=DDPStrategy(find_unused_parameters=False), deterministic=True,
                         accelerator='gpu', devices=gpus, num_nodes=nodes, max_epochs=1, logger=False)

    trainer.fit(model, data_module)

    # tensor / json case
    train_manager = FFCVPipelineManager("./data/tensor_bytes.beton",
                                        pipeline_transforms=[
                                            # tensor pipeline
                                            [NDArrayDecoder(), ToTensor()],

                                            # json
                                            [BytesDecoder()]
                                        ],
                                        ordering=OrderOption.RANDOM)

    val_manager = FFCVPipelineManager("./data/tensor_bytes.beton",
                                      pipeline_transforms=[
                                          # tensor pipeline
                                          None,

                                          # json
                                          [BytesDecoder()]
                                      ])

    data_module = FFCVDataModule(batch_size, workers, train_manager=train_manager, val_manager=val_manager,
                                 is_dist=True, seed=SEED)

    # trainer
    trainer = pl.Trainer(strategy=DDPStrategy(find_unused_parameters=False), deterministic=True,
                         accelerator='gpu', devices=gpus, num_nodes=nodes, max_epochs=1, logger=False)

    trainer.fit(model, data_module)

    if os.path.exists('./data/'):
        shutil.rmtree('./data/')

    if os.path.exists('./checkpoints/'):
        shutil.rmtree('./checkpoints/')
