from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

from setuptools import setup, find_packages

install_requires = ['recbole==1.0.1', 'torch>=1.7.0', 'scipy==1.6.0', 'numpy==1.20.0',
                    'pymoo>=0.6.1', 'numba>=0.56.4', 'xgboost>=2.1.2']

setup_requires = []

extras_require = {
    'hyperopt': ['hyperopt>=0.2.4']
}

classifiers = ["License :: OSI Approved :: MIT License"]

# Readthedocs requires Sphinx extensions to be specified as part of
# install_requires in order to build properly.
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    install_requires.extend(setup_requires)

setup(
    name='MCRecKit',
    version=
    '1.0.0',  # please remember to edit mcreckit/__init__.py in response, once updating the version
    description='MCRecKit: An Open-Source Library for Multi-Criteria Recommendations',
    long_description_content_type="text/markdown",
    url='https://github.com/irecsys/MCRecKit',
    author='Yong Zheng, David Xuejun Wang, Qin Ruan',
    author_email='DeepCARSKit@Gmail.com',
    packages=[
        package for package in find_packages()
        if package.startswith('mcreckit')
    ],
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require=extras_require,
    zip_safe=False,
    classifiers=classifiers,
)
