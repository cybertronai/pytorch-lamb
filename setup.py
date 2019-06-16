from setuptools import find_packages, setup

setup(
    name="pytorch_lamb",
    author="Ben Mann",
    version='1.0.0',
    author_email="me@benjmann.net",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    license='MIT',
    url="https://github.com/cybertronai/pytorch-lamb",
    install_requires=[
        'torch>=0.4.1',
        'tqdm',
        'tensorboardX',
        'torchvision',
    ],
)
