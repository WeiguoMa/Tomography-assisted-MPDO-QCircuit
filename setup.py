import sys

try:
    from setuptools import setup
except ImportError:
    print(
        "Please update pip, you need pip 10 or greater,\n"
        " or you need to install the PEP 518 requirements in pyproject.toml yourself",
        file=sys.stderr,
    )
    raise

from os import path

from setuptools import find_packages

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

def parse_requirements(filename):
     with open(filename, 'r') as file:
        lines = (line.strip() for line in file)
        return [line for line in lines if line and not line.startswith("#")]

requirements = parse_requirements("requirements.txt")

setup(
    name="MPDOSimulator",
    version="1.0.0",
    description="Tomography-assisted quantum circuit simulator with Matrix Product Density Operators",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Weiguo Ma",
    author_email="Weiguo.m@iphy.ac.cn",
    url="https://github.com/WeiguoMa/Tomography-assisted-MPDO-QCircuit",
    packages=find_packages(),
    package_data={
        "MPDOSimulator": ["Chi/*.mat"]
    },
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: BSD 3-Clause License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
