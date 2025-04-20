from setuptools import setup, find_packages

setup(
    name="qicklab",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "matplotlib",
        "visdom",
        "numpy",
        "tqdm",
        "scipy",
        "datetime",
        "h5py",
        "allantools",
        "math"
    ],
)
