from setuptools import setup, find_packages

setup(
    name="eqprop",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "tqdm",
        "numpy",
    ],
)
