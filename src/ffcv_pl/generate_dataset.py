from ffcv import DatasetWriter
from ffcv.fields import Field
from torch.utils.data import Dataset
from ffcv_pl.ffcv_utils.utils import field_to_str


def create_beton_wrapper(torch_dataset: Dataset, output_path: str, fields: tuple[Field, ...]) -> None:
    """
    :param torch_dataset:
    Predefined Pytorch Dataset object.
    Can have any number/type of parameters in the __init__ method.
    Constraints on the __get_item__ method:
    According to the official ffcv docs, the dataset must return a tuple object of any length.
    See for example: https://docs.ffcv.io/writing_datasets.html
    The type of the elements inside the tuple is restricted to the ffcv.fields admitted
    types: https://docs.ffcv.io/api/fields.html
    (PIL Images - RGBImageField, strings or variable length bytes for BytesField, integers for IntField,
    floats for FloatField, numpy arrays for NDArrayField, dicts for JSONField and torch tensors for TorchTensorField)

    :param output_path: complete string for the beton output path, without extension. Example: "./train" will create a
    "./train.beton" file at the end of process.

    :param fields: iterable of ffcv.fields objects specifying types that the __get_item__ method of
    the torch_dataset returns.
    For example, a torch dataset that returns a tuple: (PILImage, intlabel) in the __get_item__ method should specify an
    iterable like: (RGBImageField(), IntField) \n
    Note that some fields require one or more parameters, like shape and dtype for NDArrays and Torch Tensors.
    Check https://docs.ffcv.io/api/fields.html for a complete documentation.
    """

    # 1. format output path
    assert len(output_path) > 0, 'param: output_path cannot be an empty string'

    if not output_path.endswith('.beton'):
        output_path = f'{output_path}.beton'

    # 2. create dict of fields
    assert len(fields) > 0, '*fields parameter must be an iterable with at least one element'

    final_mapping = {}
    for i, f in enumerate(fields):
        final_mapping[f'{field_to_str(type(f))}_{i}'] = f

    # official guidelines: https://docs.ffcv.io/writing_datasets.html
    print(f'[INFO] creating ffcv dataset into file: {output_path}')
    writer = DatasetWriter(output_path, final_mapping)
    writer.from_indexed_dataset(torch_dataset)
