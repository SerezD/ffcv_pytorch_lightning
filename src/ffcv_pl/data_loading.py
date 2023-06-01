import pytorch_lightning as pl
from ffcv.fields import Field
from ffcv.loader import Loader, OrderOption
import warnings
from ffcv_pl.ffcv_utils.decoders import FFCVDecoders
from ffcv_pl.ffcv_utils.utils import field_to_str


class FFCVDataModule(pl.LightningDataModule):

    def __init__(self, batch_size: int, num_workers: int, is_dist: bool, fields: tuple[Field, ...],
                 train_file: str = None, val_file: str = None, test_file: str = None, predict_file: str = None,
                 train_decoders: FFCVDecoders = None, val_decoders: FFCVDecoders = None,
                 test_decoders: FFCVDecoders = None, predict_decoders: FFCVDecoders = None,
                 train_order: OrderOption = OrderOption.RANDOM,
                 val_order: OrderOption = OrderOption.SEQUENTIAL,
                 test_order: OrderOption = OrderOption.SEQUENTIAL,
                 predict_order: OrderOption = OrderOption.SEQUENTIAL,
                 os_cache: bool = True, seed: int = None,
                 ) -> None:
        """
        :param batch_size:
        :param num_workers:
        :param is_dist: pass true if you are using more than one gpu/node
        :param fields: iterable of ffcv.fields types specifying what the __get_item__ method of
                        the .beton dataset you are loading returns.
                        Check https://docs.ffcv.io/api/fields.html for a complete documentation of available options.
        :param train_file: path to .beton train file
        :param val_file: path to .beton validation file
        :param test_file: path to .beton test file
        :param predict_order: path to .beton predict file
        :param train_decoders: Decoder object that will be used to decode and optionally apply transforms to data.
                                If None, uses the Default Decoders (just loading with no preprocessing)
        :param val_decoders: Decoder object that will be used to decode and optionally apply transforms to data.
                                If None, uses the Default Decoders (just loading with no preprocessing)
        :param test_decoders: Decoder object that will be used to decode and optionally apply transforms to data.
                                If None, uses the Default Decoders (just loading with no preprocessing)
        :param predict_decoders: Decoder object that will be used to decode and optionally apply transforms to data.
                                If None, uses the Default Decoders (just loading with no preprocessing)
        :param train_order: ordering for data loading. (OrderOption object from ffcv.loader).
        :param val_order: ordering for data loading. (OrderOption object from ffcv.loader).
        :param test_order: ordering for data loading. (OrderOption object from ffcv.loader).
        :param predict_order: ordering for data loading. (OrderOption object from ffcv.loader).
        :param os_cache: option for the ffcv loader, depending on your dataset.
                        Read official docs: https://docs.ffcv.io/parameter_tuning.html
        :param seed: fix data loading process to ensure reproducibility
        """

        # initial condition must be satisfied
        assert not (train_file is None and val_file is None and test_file is None and predict_file is None), \
            'At least one file between train, val, test and predict must be specified'

        super().__init__()

        # assign default decoders if none
        labels = ['train', 'val', 'test', 'predict']
        decoders = [train_decoders, val_decoders, test_decoders, predict_decoders]
        files = [train_file, val_file, test_file, predict_file]

        for i, (l, d, f) in enumerate(zip(labels, decoders, files)):

            if d is None:

                decoders[i] = FFCVDecoders()

                if f is not None:
                    message = f'you specified a {l} file but not the {l} decoder. ' \
                              f'Running with default Decoding Pipeline'
                    warnings.warn(message)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.is_dist = is_dist
        self.os_cache = os_cache
        self.fields = [field_to_str(f) for f in fields]

        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.predict_file = predict_file

        self.train_decoders = decoders[0]
        self.val_decoders = decoders[1]
        self.test_decoders = decoders[2]
        self.predict_decoders = decoders[3]

        self.train_pipeline = None
        self.val_pipeline = None
        self.test_pipeline = None
        self.predict_pipeline = None

        self.train_order = train_order
        self.val_order = val_order
        self.test_order = test_order
        self.predict_order = predict_order

    def prepare_data(self) -> None:
        """
        This method is used to define the processes that are meant to be performed by only one GPU.
        It’s usually used to handle the task of downloading the data.
        """
        pass

    def setup(self, stage: str) -> None:

        pipeline = {}

        if stage == 'fit':

            self.train_pipeline = {}
            self.val_pipeline = {}

            for i, f in enumerate(self.fields):
                if f == 'image':
                    t_value = self.train_decoders.image_transforms
                    v_value = self.val_decoders.image_transforms
                elif f == 'bytes':
                    t_value = self.train_decoders.bytes_transforms
                    v_value = self.val_decoders.bytes_transforms
                elif f == 'int':
                    t_value = self.train_decoders.int_transforms
                    v_value = self.val_decoders.int_transforms
                elif f == 'float':
                    t_value = self.train_decoders.float_transforms
                    v_value = self.val_decoders.float_transforms
                elif f == 'array':
                    t_value = self.train_decoders.array_transforms
                    v_value = self.val_decoders.array_transforms
                elif f == 'json':
                    t_value = self.train_decoders.json_transforms
                    v_value = self.val_decoders.json_transforms
                elif f == 'tensor':
                    t_value = self.train_decoders.tensor_transforms
                    v_value = self.val_decoders.tensor_transforms
                else:
                    t_value = None
                    v_value = None

                if t_value is not None:
                    self.train_pipeline[f'{f}_{i}'] = t_value
                if v_value is not None:
                    self.val_pipeline[f'{f}_{i}'] = v_value

        elif stage == 'train':

            for i, f in enumerate(self.fields):
                if f == 'image':
                    value = self.train_decoders.image_transforms
                elif f == 'bytes':
                    value = self.train_decoders.bytes_transforms
                elif f == 'int':
                    value = self.train_decoders.int_transforms
                elif f == 'float':
                    value = self.train_decoders.float_transforms
                elif f == 'array':
                    value = self.train_decoders.array_transforms
                elif f == 'json':
                    value = self.train_decoders.json_transforms
                elif f == 'tensor':
                    value = self.train_decoders.tensor_transforms
                else:
                    value = None

                if value is not None:
                    pipeline[f'{f}_{i}'] = value

            self.train_pipeline = pipeline

        elif stage == 'validate':

            for i, f in enumerate(self.fields):
                if f == 'image':
                    value = self.val_decoders.image_transforms
                elif f == 'bytes':
                    value = self.val_decoders.bytes_transforms
                elif f == 'int':
                    value = self.val_decoders.int_transforms
                elif f == 'float':
                    value = self.val_decoders.float_transforms
                elif f == 'array':
                    value = self.val_decoders.array_transforms
                elif f == 'json':
                    value = self.val_decoders.json_transforms
                elif f == 'tensor':
                    value = self.val_decoders.tensor_transforms
                else:
                    value = None

                if value is not None:
                    pipeline[f'{f}_{i}'] = value

            self.val_pipeline = pipeline

        elif stage == 'test':

            for i, f in enumerate(self.fields):
                if f == 'image':
                    value = self.test_decoders.image_transforms
                elif f == 'bytes':
                    value = self.test_decoders.bytes_transforms
                elif f == 'int':
                    value = self.test_decoders.int_transforms
                elif f == 'float':
                    value = self.test_decoders.float_transforms
                elif f == 'array':
                    value = self.test_decoders.array_transforms
                elif f == 'json':
                    value = self.test_decoders.json_transforms
                elif f == 'tensor':
                    value = self.test_decoders.tensor_transforms
                else:
                    value = None

                if value is not None:
                    pipeline[f'{f}_{i}'] = value

            self.test_pipeline = pipeline

        elif stage == 'predict':

            for i, f in enumerate(self.fields):
                if f == 'image':
                    value = self.predict_decoders.image_transforms
                elif f == 'bytes':
                    value = self.predict_decoders.bytes_transforms
                elif f == 'int':
                    value = self.predict_decoders.int_transforms
                elif f == 'float':
                    value = self.predict_decoders.float_transforms
                elif f == 'array':
                    value = self.predict_decoders.array_transforms
                elif f == 'json':
                    value = self.predict_decoders.json_transforms
                elif f == 'tensor':
                    value = self.predict_decoders.tensor_transforms
                else:
                    value = None

                if value is not None:
                    pipeline[f'{f}_{i}'] = value

            self.predict_pipeline = pipeline

        else:
            pass

    def train_dataloader(self):
        if self.train_file is None:
            pass
        return Loader(self.train_file, batch_size=self.batch_size, num_workers=self.num_workers, os_cache=self.os_cache,
                      order=self.train_order, pipelines=self.train_pipeline, distributed=self.is_dist, seed=self.seed)

    def val_dataloader(self):
        if self.val_file is None:
            pass
        return Loader(self.val_file, batch_size=self.batch_size, num_workers=self.num_workers, os_cache=self.os_cache,
                      order=self.val_order, pipelines=self.val_pipeline, distributed=self.is_dist, seed=self.seed)

    def test_dataloader(self):
        if self.test_file is None:
            pass
        return Loader(self.test_file, batch_size=self.batch_size, num_workers=self.num_workers, os_cache=self.os_cache,
                      order=self.test_order, pipelines=self.test_pipeline, distributed=self.is_dist, seed=self.seed)

    def predict_dataloader(self):
        if self.predict_file is None:
            pass
        return Loader(self.predict_file, batch_size=self.batch_size, num_workers=self.num_workers,
                      os_cache=self.os_cache, order=self.predict_order, pipelines=self.predict_pipeline,
                      distributed=self.is_dist, seed=self.seed)
