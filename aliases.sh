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
conda install torchio

# Step 1: Download the TCIA manifest (a copy is included)
# Step 2: Set the target of the NBIA data retriever: 'C:\Users\justin\PycharmProjects\cpsc464-group2\image_data'
# Step 3: intermittently use remove_invalid.py to find and remove invalid images
# Step 4: intermittently use filtering.py to find and remove unnecessary images
# Step 5: finally, use downscale.py to standardize the resolution of each dicom file
