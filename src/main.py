import ffcv.fields
import pytorch_lightning as pl
import torch
from ffcv.fields.rgb_image import RandomResizedCropRGBImageDecoder
from ffcv.transforms import ToTensor, ToTorchImage
from pytorch_lightning.strategies.ddp import DDPStrategy

from torch import nn
from torch.optim import Adam
from torchvision.transforms import RandomHorizontalFlip

from ffcv_pl.data_loading import FFCVDataModule
from ffcv_pl.ffcv_utils.augmentations import DivideImage255
from ffcv_pl.ffcv_utils.decoders import FFCVDecoders


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):

    def __init__(self):

        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(32 * 32 * 3, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 32 * 32 * 3))

    def training_step(self, batch, batch_idx):

        x, y = batch

        b, c, h, w = x.shape
        x = x.reshape(b, -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == '__main__':

    SEED = 1234

    pl.seed_everything(SEED, workers=True)

    batch_size = 16
    gpus = 2
    nodes = 1
    workers = 8

    # define model
    model = LitAutoEncoder()

    # trainer
    trainer = pl.Trainer(strategy=DDPStrategy(find_unused_parameters=False), deterministic=True,
                         accelerator='gpu', devices=gpus, num_nodes=nodes, max_epochs=5, logger=False)

    # dataset and fields
    train_path = './src/image_label.beton'  # created with the `create_beton_wrapper` method
    fields = (ffcv.fields.RGBImageField, ffcv.fields.IntField)  # return type of the __get_item__ method

    # use the FFCVDecoders object to specify the transform for the Image field (keeping the default for Int).
    # these transforms will be given to the Loading pipeline https://docs.ffcv.io/making_dataloaders.html
    decoders = FFCVDecoders(image_transforms=[RandomResizedCropRGBImageDecoder((32, 32)), ToTensor(),
                                              ToTorchImage(), DivideImage255(dtype=torch.float32),
                                              RandomHorizontalFlip(p=0.5)])

    # create the PL Data Module with FFCV Loader
    data_module = FFCVDataModule(batch_size, workers, gpus > 1 or nodes > 1, fields=fields,
                                 train_file=train_path, train_decoders=decoders, seed=SEED)

    # pass data module to the fit method, as usual
    trainer.fit(model, data_module)
