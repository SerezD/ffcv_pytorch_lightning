from ffcv import DatasetWriter
from torch.utils.data import Dataset
import os

from ffcv_pl.ffcv_utils.utils import field_to_str, obj_to_field


def create_beton_wrapper(torch_dataset: Dataset, output_path: str) -> None:
    """
    :param torch_dataset: Pytorch Dataset object (https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

    The dataset can have any number/type of parameters in the __init__ method.

    Constraints on the __get_item__ method:
        According to the official ffcv docs, the dataset must return a tuple object of any length.
        See for example: https://docs.ffcv.io/writing_datasets.html

        The type of the elements inside the tuple is restricted to the ffcv.fields admitted types:
        https://docs.ffcv.io/api/fields.html

        (PIL Images - RGBImageField, integers for IntField, floats for FloatField, numpy arrays for NDArrayField,
        dicts for JSONField, torch tensors for TorchTensorField and
        1D uint8 numpy array of variable length for BytesField)

    :param output_path: desired path for .beton output file, "/" separated. E.g. "./my_dataset.beton"

    """

    # 1. format output path
    assert len(output_path) > 0, 'param: output_path cannot be an empty string'

    if not output_path.endswith('.beton'):
        output_path = f'{output_path}.beton'

    # find dir
    dir_name = '/'.join(output_path.split('/')[:-1])
    if not os.path.exists(dir_name):
        print(f'[INFO] Creating output folder: {dir_name}')
        os.makedirs(dir_name)

    # 2. check that dataset __get_item__ returns a tuple and get fields.
    tuple_obj = torch_dataset[0]

    if not isinstance(tuple_obj, tuple):
        raise AttributeError("According to the official ffcv docs, the dataset must return a tuple object. "
                             "See for example: https://docs.ffcv.io/writing_datasets.html")

    fields = []
    for obj in tuple_obj:
        fields.append(obj_to_field(obj))

    # 2. create dict of fields
    final_mapping = {}
    for i, f in enumerate(fields):
        final_mapping[f'{field_to_str(type(f))}_{i}'] = f

    # official guidelines: https://docs.ffcv.io/writing_datasets.html
    print(f'[INFO] creating ffcv dataset into file: {output_path}')
    print(f'[INFO] number of items: {len(torch_dataset)}')
    print(f'[INFO] ffcv fields of items: {fields}')

    writer = DatasetWriter(output_path, final_mapping)
    writer.from_indexed_dataset(torch_dataset)

    print(f'[INFO] Done.')
    print(f'#' * 30)
