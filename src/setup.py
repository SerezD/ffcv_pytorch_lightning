from setuptools import setup

long_description = """
                   Version Updates: allow optional arguments for FFCVDataModule and create_beton_wrapper.\n
                   For example usages/installation, 
                   check [github project](https://github.com/SerezD/ffcv_pytorch_lightning)
                   """

setup(name='ffcv_pl',
      version='0.3.2',
      packages=['ffcv_pl', 'ffcv_pl.ffcv_utils'],
      description='manage fast data loading with ffcv and pytorch lightning',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/SerezD/ffcv_pytorch_lightning',
      author='DSerez',
      license='MIT',
      zip_safe=False)
