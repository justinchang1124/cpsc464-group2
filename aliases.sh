# The purpose of this file is to store useful commands
# Project was developed in pycharm community

conda info --envs
# conda remove --name cpsc464-group2
# you can always make the env again from settings -> python interpreter -> gear -> add
# note: if installation is super slow, try restarting pycharm
conda install keras
conda install tensorflow
conda install matplotlib.pylab
conda install opencv
conda install scikit-learn
pip install pydicom

# target of NBIA data retriever: C:\Users\justin\PycharmProjects\cpsc464-group2\image_data
# workflow:
# - once a folder is downloaded by the NBIA data retriever, decrease the size of the contents
# - - only keep the relevant passes
# - - only keep the relevant images

# STEPS:
# - get rid of segmentation files manually
# - get rid of images that are too 'far out' (not correctly bounded on each side)
# - make training / testing / validation on a per-sample basis

# note: you can edit the .tcia manifest, current split:
# P1: 0-25%
# P2: 50%-100%
# P3: 25%-50%


# idea: simply sample the middle image from each scan