#!/bin/bash
set -x

apt-get upgrade -y && apt-get update -y && apt-get install unzip -y
pip install --upgrade termcolor matplotlib opencv-python scikit-learn
cd datasets
wget -q --show-progress "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
tar -zxf CUB_200_2011.tgz