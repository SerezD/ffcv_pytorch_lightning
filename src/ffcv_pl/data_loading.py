import pytorch_lightning as pl
from ffcv.loader import Loader
from ffcv_pl.ffcv_utils.utils import FFCVPipelineManager


class FFCVDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int, num_workers: int, is_dist: bool, train_manager: FFCVPipelineManager = None,
                 val_manager: FFCVPipelineManager = None, test_manager: FFCVPipelineManager = None,
                 predict_manager: FFCVPipelineManager = None, os_cache: bool = True, seed: int = None,
                 ) -> None:
        """
        Define PL DataModule (https://lightning.ai/docs/pytorch/stable/data/datamodule.html) object using
        FFCV Loader (https://docs.ffcv.io/making_dataloaders.html)
        :param batch_size: batch_size for loader objects
        :param num_workers: num workers for loader objects
        :param is_dist: pass true if using more than one gpu/node
        :param train_manager: manager for the training data, ignore if not loading train data
        :param val_manager: manager for the validation data, ignore if not loading validation data
        :param test_manager: manager for the test data, ignore if not loading test data
        :param predict_manager: manager for the predict data, ignore if not loading predict data
        :param os_cache: option for the ffcv loader, depending on your dataset.
                        Read official docs: https://docs.ffcv.io/parameter_tuning.html
        :param seed: fix data loading process to ensure reproducibility
        """

        # initial condition must be satisfied
        if train_manager is None and val_manager is None and test_manager is None and predict_manager is None:
            raise AttributeError('At least one file between train, val, test and predict manager must be specified')

        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.os_cache = os_cache

        self.is_dist = is_dist

        self.train_manager = train_manager
        self.val_manager = val_manager
        self.test_manager = test_manager
        self.predict_manager = predict_manager

    def prepare_data(self) -> None:
        """
        This method is used to define the processes that are meant to be performed by only one GPU.
        It’s usually used to handle the task of downloading the data.
        """
        pass

    def setup(self, stage: str) -> None:
        pass

    def train_dataloader(self):

        if self.train_manager is not None:
            return Loader(self.train_manager.file_path,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          os_cache=self.os_cache,
                          order=self.train_manager.ordering,
                          pipelines=self.train_manager.pipeline,
                          distributed=self.is_dist,
                          seed=self.seed)

    def val_dataloader(self):
        if self.val_manager is not None:
            return Loader(self.val_manager.file_path,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          os_cache=self.os_cache,
                          order=self.val_manager.ordering,
                          pipelines=self.val_manager.pipeline,
                          distributed=self.is_dist,
                          seed=self.seed)

    def test_dataloader(self):
        if self.test_manager is not None:
            return Loader(self.test_manager.file_path,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          os_cache=self.os_cache,
                          order=self.test_manager.ordering,
                          pipelines=self.test_manager.pipeline,
                          distributed=self.is_dist,
                          seed=self.seed)

    def predict_dataloader(self):
        if self.predict_manager is not None:
            return Loader(self.predict_manager.file_path,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          os_cache=self.os_cache,
                          order=self.predict_manager.ordering,
                          pipelines=self.predict_manager.pipeline,
                          distributed=self.is_dist,
                          seed=self.seed)
