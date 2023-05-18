from ffcv.fields import Field, RGBImageField, BytesField, IntField, FloatField, NDArrayField, JSONField, \
    TorchTensorField


def field_to_str(f: Field) -> dict:
    mapping = {RGBImageField: "image",
               BytesField: "bytes",
               IntField: "int",
               FloatField: "float",
               NDArrayField: "array",
               JSONField: "json",
               TorchTensorField: "tensor"}
    return mapping[f]
