import pytorch_lightning as pl
import torch
from pytorch_lightning.strategies.ddp import DDPStrategy

from torch import nn
from torch.optim import Adam

from ffcv_pl.datasets.image import ImageDataModule


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):

    def __init__(self):

        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(256 * 256 * 3, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 256 * 256 * 3))

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

    dataset = 'cub2002011'
    image_size = 256
    batch_size = 16
    train_folder = f'/media/dserez/datasets/{dataset}/train.beton'
    val_folder = f'/media/dserez/datasets/{dataset}/test.beton'

    gpus = 2
    workers = 8

    # define model
    model = LitAutoEncoder()

    # trainer
    trainer = pl.Trainer(strategy=DDPStrategy(find_unused_parameters=False), deterministic=True,
                         accelerator='gpu', devices=gpus, num_nodes=1, max_epochs=5)

    # Note: set is_dist True if you are using DDP and more than one GPU
    data_module = ImageDataModule(train_folder, val_folder, val_folder, image_size, torch.float32, batch_size,
                                  num_workers=1, is_dist=gpus > 1, seed=SEED)

    trainer.fit(model, data_module)
