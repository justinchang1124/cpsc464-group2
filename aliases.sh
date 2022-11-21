# The purpose of this file is to store useful commands
# Project was developed in pycharm community

conda info --envs
# conda env remove --name cpsc464-group2
# you can always make the env again from settings -> python interpreter -> gear -> add
# note: if installation is super slow, try restarting pycharm

# the hefty machine learning libraries
conda install keras
conda install tensorflow
conda install torchvision -c pytorch
conda install tqdm

# other
conda install pandas
conda install -c conda-forge matplotlib
conda install opencv
conda install scikit-learn
pip install pydicom

# Step 1: split the .tcia manifest, perhaps like below:
# P1: 0-25%
# P2: 50%-100%
# P3: 25%-50%

# target of NBIA data retriever: C:\Users\justin\PycharmProjects\cpsc464-group2\image_data

# For each part:
# Step 2: use test_dcm_utils.py to find and remove unopenable images
# Step 3: use filtering.py to find and remove unnecessary images
# Step 4: use downsample.py to standardize the resolution of each dicom file
