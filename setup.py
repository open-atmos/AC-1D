import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="LIIFE-CM",
    version="0.0.1",
    url="https://github.com/isilber/cld_INP_1D_model",
    license='MIT',

    author="Israel Silber",
    author_email="ixs34@psu.edu",

    description="LES_informed 1D model for the evaluation of different ice formation mechanisms from literature",
    long_description=read("README.rst"),
    include_package_data=True,
    packages=find_packages(exclude=('tests',)),

    install_requires=[],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
