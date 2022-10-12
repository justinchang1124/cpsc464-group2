import pydicom as dicom
import os
import matplotlib
matplotlib.use("Qt5Agg")
import glob
import re
import numpy as np

abs_proj_path = 'C:/Users/justin/PycharmProjects/breast_cancer_mri_conda'
data_path = 'image_data/manifest-1654812109500/Duke-Breast-Cancer-MRI'
abs_data_path = '{}/{}'.format(abs_proj_path, data_path)

all_data_files = []
os.chdir(abs_data_path)
for filename in glob.iglob('**/*', recursive=True):
    all_data_files.append(filename)

dcm_data_files = []
for filename in all_data_files:
    if re.search('.dcm$', filename):
        dcm_data_files.append(filename.split("\\"))

dcm_data_files_mat = np.array(dcm_data_files)
examp_dcm_loc_mat = dcm_data_files_mat[0]


def read_dcm_loc(dcm_loc_mat, base_path=abs_data_path):
    joined_path = '\\'.join(dcm_loc_mat)
    return dicom.dcmread('{}/{}'.format(base_path, joined_path))

x = read_dcm_loc(examp_dcm_loc_mat)
matplotlib.pylab.imshow(x.pixel_array)



# we want to create a dir struct as follows:
# base_path
# - sample 1 path name
# -



for folder1 in os.listdir(base_path):
    subfolder = os.listdir("{}/{}".format(base_path, folder))
    print(subfolder)

# ds = dicom.dcmread(base_path)

# pixel_array_numpy = ds.pixel_array