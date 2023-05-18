
class FFCVDecoders:

    def __init__(self, image_transforms: list = None, bytes_transforms: list = None, int_transforms: list = None,
                 float_transforms: list = None, array_transforms: list = None, json_transforms: list = None,
                 tensor_transforms: list = None):
        """
        Specify a List of operations to perform for each of the ffcv fields that you want to use:
        https://docs.ffcv.io/api/fields.html

        The non specified arguments will be set to the base decoder for that field:
        https://docs.ffcv.io/api/decoders.html

        """

        self.image_transforms = image_transforms
        self.bytes_transforms = bytes_transforms
        self.int_transforms = int_transforms
        self.float_transforms = float_transforms
        self.array_transforms = array_transforms
        self.json_transforms = json_transforms
        self.tensor_transforms = tensor_transforms
