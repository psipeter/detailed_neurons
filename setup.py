#!/usr/bin/env python

import imp
import io
import os
import sys

try:
    from setuptools import find_packages, setup
except ImportError:
    raise ImportError(
        "'setuptools' is required but not installed. To install it, "
        "follow the instructions at "
        "https://pip.pypa.io/en/stable/installing/#installing-with-get-pip-py")
    
name = 'detailed_neurons'
root = os.path.dirname(os.path.realpath(__file__))
install_requires = ["numpy>=1.11"]

def read(*filenames, **kwargs):
    encoding = kwargs.get("encoding", "utf-8")
    sep = kwargs.get("sep", "\n")
    buf = []
    for filename in filenames:
        with io.open(filename, encoding=encoding) as f:
            buf.append(f.read())
    return sep.join(buf)

setup(
    name=name,
    version='1.0',
    description="Methods for training nengo networks containing detailed neuron models",
    url='https://github.com/psipeter/detailed_neurons',
    author='Peter Duggins',
    author_email='psipeter@gmail.com',
    packages=['detailed_neurons'],
    long_description=read("README.rst"),
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False
)
