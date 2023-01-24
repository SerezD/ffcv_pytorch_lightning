import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

from torch import nn
from torch.optim import Adam

from src.datasets.datamodule import ImageDataModule


# define the LightningModule
class LitAutoEncoder(pl.LightningModule):

    def __init__(self):

        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
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

    SEED = 4219

    pl.seed_everything(SEED, workers=True)

    dataset = 'dummy'
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
    data_module = ImageDataModule(train_folder, val_folder, image_size, batch_size, workers,
                                  is_dist=gpus > 1, seed=SEED)

    trainer.fit(model, data_module)
