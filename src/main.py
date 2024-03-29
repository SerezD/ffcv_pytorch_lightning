import pytorch_lightning as pl
import torch
from ffcv.fields.basics import IntDecoder
from ffcv.fields.rgb_image import RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
from ffcv.loader import OrderOption
from ffcv.transforms import ToTensor, ToTorchImage
from pytorch_lightning.strategies.ddp import DDPStrategy

from torch import nn
from torch.optim import Adam
from torchvision.transforms import RandomHorizontalFlip

from ffcv_pl.data_loading import FFCVDataModule
from ffcv_pl.ffcv_utils.augmentations import DivideImage255

from ffcv_pl.ffcv_utils.utils import FFCVPipelineManager


# define the LightningModule
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


def main():

    seed = 1234

    pl.seed_everything(seed, workers=True)

    batch_size = 16
    gpus = 2
    nodes = 1
    workers = 8

    # image label dataset
    train_manager = FFCVPipelineManager("./data/image_label.beton",  # previously defined using dataset_creation.py
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
                                        ordering=OrderOption.RANDOM)  # random ordering for training

    val_manager = FFCVPipelineManager("./data/image_label.beton",
                                      pipeline_transforms=[

                                          # image pipeline (different from train)
                                          [CenterCropRGBImageDecoder((32, 32), ratio=1.),
                                           ToTensor(),
                                           ToTorchImage(),
                                           DivideImage255(dtype=torch.float32)],

                                          # label (int) pipeline
                                          None  # if None, uses default
                                      ],
                                      ordering=OrderOption.SEQUENTIAL)  # sequential ordering for validation

    # datamodule creation
    # ignore test and predict steps, since managers are not defined.
    data_module = FFCVDataModule(batch_size, workers, train_manager=train_manager, val_manager=val_manager,
                                 is_dist=True, seed=seed)

    # define model
    model = LitAutoEncoder()

    # trainer
    trainer = pl.Trainer(strategy=DDPStrategy(find_unused_parameters=False), deterministic=True,
                         accelerator='gpu', devices=gpus, num_nodes=nodes, max_epochs=5, logger=False)

    # start training!
    trainer.fit(model, data_module)


if __name__ == '__main__':

    main()
