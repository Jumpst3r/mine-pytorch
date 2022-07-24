import setuptools
import sys
if sys.version_info[1] == 10:
    torchwheel = "http://download.pytorch.org/whl/cpu/torch-1.11.0%2Bcpu-cp310-cp310-linux_x86_64.whl"
if sys.version_info[1] == 9:
    torchwheel = "http://download.pytorch.org/whl/cpu/torch-1.11.0%2Bcpu-cp39-cp39-linux_x86_64.whl"

setuptools.setup(
    name="mine", # Replace with your own username
    version="0.0.1",
    author="",
    author_email="",
    description="An implementation of the MINE algorithm in Pytorch",
    long_description="",
    long_description_content_type="",
    url="https://github.com/gtegner/mine-pytorch",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires = [
        f'torch @ {torchwheel}',
        'numpy'
    ]
)