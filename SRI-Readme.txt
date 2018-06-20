To upload new versions of this library to PyPi:

1.0 What is in this README?:
============================
This README explains how to build and upload the SRI Fork of TPOT to the PyPi library


2.0 Version:
============
Date instructions updated: June 13th 2018
Version: v2018.6.5 (To match the Summer 2018 eval version)


3.0 PyPi:
=========
Instructions on getting started with PyPi are here:

    > https://packaging.python.org/tutorials/packaging-projects/


4.0 Instructions on uploading new versions of SRITPOT to PyPi:
==============================================================

1. Get credentials to a PyPi account

2. Checkout the sri fork of tpot:

    > git clone https://github.com/daraghhartnett/tpot
    > cd tpot

3. Build the distribution:

    * See the packaging-projects instructions in 3.0 if you dont have setuptools, wheel or twine installed before
      you try to upload.
    * Note that if a version already exists on PyPi you will need to roll the version in ./tpot/_version.py
      and the log message in base.py before you try to upload.
    > python setup.py sdist bdist_wheel
    > ~/.local/bin/twine upload dist/*

4. To install the SRI fork of TPOT do the following:

   > pip install SRITPOT

   * or the following if you already have it installed and want the latest version:
   > pip uninstall SRITPOT
   > pip install --no-cache-dir --upgrade SRITPOT
   * And for a specific version:
   > pip install --no-cache-dir SRITPOT==0.9.6
