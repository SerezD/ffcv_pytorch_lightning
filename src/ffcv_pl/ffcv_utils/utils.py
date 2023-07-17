from typing import Any

import numpy as np
import torch
from PIL.Image import Image
from ffcv.fields import Field, RGBImageField, BytesField, IntField, FloatField, NDArrayField, JSONField, \
    TorchTensorField

from ffcv.loader import OrderOption
from ffcv.reader import Reader


def field_to_str(f: Field) -> str:
    mapping = {RGBImageField: "image",
               BytesField: "bytes",
               IntField: "int",
               FloatField: "float",
               NDArrayField: "array",
               JSONField: "json",
               TorchTensorField: "tensor"}
    return mapping[f]


def obj_to_field(obj: Any) -> Field:

    if isinstance(obj, Image):
        return RGBImageField(write_mode="jpg")

    elif isinstance(obj, int):
        return IntField()

    elif isinstance(obj, float):
        return FloatField()

    elif isinstance(obj, np.ndarray) and not isinstance(obj[0], np.uint8):
        return NDArrayField(obj.dtype, obj.shape)

    elif isinstance(obj, np.ndarray) and isinstance(obj[0], np.uint8):
        return BytesField()

    elif isinstance(obj, dict):
        return JSONField()

    elif isinstance(obj, torch.Tensor):
        return TorchTensorField(obj.dtype, obj.shape)

    else:
        raise AttributeError(f"FFCV dataset can not manage {type(obj)} objects")


class FFCVPipelineManager:

    def __init__(self, file_path: str, pipeline_transforms: list[list], ordering: OrderOption = OrderOption.SEQUENTIAL):
        """
        :param file_path: path to the .beton file that needs to be loaded.
        :param pipeline_transforms: similar to FFCV Pipelines: https://docs.ffcv.io/making_dataloaders.html
                                    each item is a list of operations to perform on the specific object returned by the
                                    dataset. Note that order matters.
                                    If one item is None, will apply the default pipeline.
        :param ordering: order option for this pipeline, following FFCV specs on Dataset Ordering.
        """

        self.file_path = file_path
        self.ordering = ordering

        self.pipeline = {}
        field_names = Reader(file_path).field_names

        if len(field_names) != len(pipeline_transforms):
            raise AttributeError(f'Passed pipeline_transforms object must include transforms for {len(field_names)} '
                                 f'items, {len(pipeline_transforms)} specified.')

        for name, transforms in zip(field_names, pipeline_transforms):

            if transforms is not None:
                self.pipeline[name] = transforms
