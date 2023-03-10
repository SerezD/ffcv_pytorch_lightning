from setuptools import setup
from pathlib import Path

# set readme
this_directory = Path(__file__).parent
long_description = (this_directory / "../readme.md").read_text()

setup(name='ffcv_pl',
      version='0.1.3',
      description='manage fast data loading with ffcv and pytorch lightning',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/SerezD/ffcv_pytorch_lightning',
      author='DSerez',
      license='MIT',
      zip_safe=False)
