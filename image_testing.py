import os
import pydicom as dicom
import matplotlib
import glob
import numpy as np
from matplotlib.animation import FuncAnimation

matplotlib.use("TkAgg") # Qt5Agg is terrible
import matplotlib.pyplot as plt


# recursively gets the RELATIVE locations of all DCM files in abs_base_path/folder
# abs_dir_path: an absolute path to a directory
def dcm_dir_list(abs_dir_path):
    if not (os.path.isabs(abs_dir_path) and os.path.isdir(abs_dir_path)):
        raise ValueError("Not an absolute value path!")
    cur_cwd = os.getcwd()
    os.chdir(abs_dir_path)
    result = list(glob.iglob('**/*.dcm', recursive=True))
    os.chdir(cur_cwd)
    return result


# opens a specific DCM image
def open_dcm_image(abs_dcm_file):
    if not (os.path.isabs(abs_dcm_file) and os.path.isfile(abs_dcm_file)):
        raise ValueError("Not an absolute file path!")
    return dicom.dcmread(abs_dcm_file).pixel_array


# opens all DCM images in a folder
def open_dcm_images(abs_dir_path):
    dcm_filenames = dcm_dir_list(abs_dir_path)
    n_dcm_files = len(dcm_filenames)
    result = [None] * n_dcm_files
    for i in range(n_dcm_files):
        abs_dcm_file = os.path.join(abs_dir_path, dcm_filenames[i])
        result[i] = open_dcm_image(abs_dcm_file)
    return result


# animates a list of DCM images
def animate_dcm_images(dcm_images):
    if not isinstance(dcm_images, list):
        raise ValueError("Images must be provided as a list!")

    n = len(dcm_images)
    if n < 1:
        raise ValueError("Animations require a positive number of frames!")

    anim_running = True
    current_frame = -1

    fig = plt.figure()
    im = plt.imshow(dcm_images[0], cmap=plt.cm.gray, vmin=0, vmax=255)

    def update(i):
        nonlocal current_frame
        current_frame = i
        im.set_array(dcm_images[i])
        return im,

    def on_click(event):
        nonlocal anim_running
        if anim_running:
            anim.event_source.stop()
            print("Latest Frame (-1 invalid): {}".format(current_frame))
            anim_running = False
        else:
            anim.event_source.start()
            anim_running = True

    # 17 milliseconds per frame yields 60 FPS
    fig.canvas.mpl_connect('button_press_event', on_click)
    anim = FuncAnimation(plt.gcf(), update, frames=n, interval=17, blit=True)
    return anim


abs_proj_path = 'C:/Users/justin/PycharmProjects/cpsc464-group2'
data_path = 'image_data/manifest-1654812109500/Duke-Breast-Cancer-MRI'
abs_data_path = os.path.join(abs_proj_path, data_path)
dcm_files = dcm_dir_list(abs_data_path)
n_dcm_files = len(dcm_files)

dcm_files_data = []
for file in dcm_files:
    dcm_files_data.append(file.split('\\'))

dcm_studies = [None] * n_dcm_files
for i in range(n_dcm_files):
    dcm_studies[i] = os.path.join(*dcm_files_data[i][:-1]) # splat
dcm_studies = sorted(list(set(dcm_studies)))

# for dcm_study in dcm_studies:
#     print(dcm_study)

for i in range(6):
    dcm_study_examp = dcm_studies[i]
    dcm_images_examp = open_dcm_images(os.path.join(abs_data_path, dcm_study_examp))
    x1 = animate_dcm_images(dcm_images_examp)
    print("current study (close to continue): {}".format(dcm_study_examp))
    plt.show()


# Breast_MRI_001\01-01-1990-NA-MRI BREAST BILATERAL WWO-97538\11.000000-ax dyn 3rd pass-41458
#


# dcm_study_examp = dcm_studies[0]
# dcm_images_examp = open_dcm_images(os.path.join(abs_data_path, dcm_study_examp))
# x1 = animate_dcm_images(dcm_images_examp[0:80])
# plt.show()




# def get_sample_from_dcm_mat(dcm_loc_mat, samp_name):
#     num_matches = 0
#     for i in range(dcm_loc_mat.shape[0]):
#         if dcm_loc_mat[i][0] == samp_name:
#             num_matches += 1
#
#     np_copy = np.empty(shape=(num_matches, dcm_loc_mat.shape[1]), dtype=dcm_loc_mat.dtype)
#     num_matches = 0
#     for i in range(dcm_loc_mat.shape[0]):
#         if dcm_loc_mat[i][0] == samp_name:
#             np_copy[num_matches] = dcm_loc_mat[i]
#             num_matches += 1
#
#     return np_copy



# goal:
# 1. filter down a DCM image set
# 2. filter down

