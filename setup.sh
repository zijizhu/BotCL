set -x

apt-get upgrade -y && apt-get update -y && apt-get install unzip -y
pip install --upgrade termcolor matplotlb opencv-python scikit-learn
cd datasets
wget -O "CUB-200-2011.zip" -q --show-progress "https://www.dropbox.com/scl/fi/zm7uisd9lop0j9uzok1it/CUB-200-2011.zip?rlkey=f4ip367m64lzxy4daqgsp6ilq&dl=0"
unzip -qq CUB-200-2011.zip