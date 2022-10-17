from setuptools import setup


with open('requirements.txt') as f:
    packages = f.read().splitlines()

setup(
    name='scgan',
    packages=['scgan', 'scgan.utils', 'scgan.data_loader',
              'scgan.losses_and_metrics', 'scgan.models',
              'scgan.models.model_utils',
              'scgan.models.model_utils.sync_batchnorm'],
    install_requires=packages
)
