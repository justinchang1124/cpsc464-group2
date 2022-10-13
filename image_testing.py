import pydicom as dicom
import os
import matplotlib
import glob
import re
import numpy as np
from matplotlib.animation import FuncAnimation

matplotlib.use("TkAgg") # Qt5Agg is terrible
import matplotlib.pyplot as plt


abs_proj_path = 'C:/Users/justin/PycharmProjects/cpsc464-group2'
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


def get_sample_from_dcm_mat(dcm_loc_mat, samp_name):
    num_matches = 0
    for i in range(dcm_loc_mat.shape[0]):
        if dcm_loc_mat[i][0] == samp_name:
            num_matches += 1

    np_copy = np.empty(shape=(num_matches, dcm_loc_mat.shape[1]), dtype=dcm_loc_mat.dtype)
    num_matches = 0
    for i in range(dcm_loc_mat.shape[0]):
        if dcm_loc_mat[i][0] == samp_name:
            np_copy[num_matches] = dcm_loc_mat[i]
            num_matches += 1

    return np_copy


def read_dcm_loc(dcm_loc_arr, base_path=abs_data_path):
    joined_path = '\\'.join(dcm_loc_arr)
    return dicom.dcmread('{}/{}'.format(base_path, joined_path))


def go_through_images(dcm_loc_mat):
    n = dcm_loc_mat.shape[0]

    if n < 1:
        return

    dcms = []
    for i in range(n):
        px_array = read_dcm_loc(dcm_loc_mat[i]).pixel_array
        for j in range(i):
            px_array[j % px_array.shape[0] % 2][j % px_array.shape[1]] = 122
        dcms.append(px_array)

    im = plt.imshow(dcms[0], cmap=plt.cm.gray)
    print_per = 10

    def update(i):
        # time_text.set_text(x='time = %.1d' % i)
        im.set_array(dcms[i])
        if i % print_per == 0:
            print("Current Frame: {}".format(i))
        return im,

    return FuncAnimation(plt.gcf(), update, frames=n, interval=20, blit=True)



dex = get_sample_from_dcm_mat(dcm_data_files_mat, "Breast_MRI_001")
x2 = go_through_images(dex[0:140])
plt.show()


