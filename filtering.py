import dcm_utils
import os
import re
import numpy as np


def id_of_file_slice(abs_dcm_file):
    truncate_left = re.search("[0-9]+[.]dcm$", abs_dcm_file)
    truncate_right = re.search("[0-9]+", truncate_left.group(0))
    return int(truncate_right.group(0))


def filter_file_slice(abs_dcm_file, slice_min, slice_max):
    dcm_utils.validate_le(slice_min, slice_max)
    slice_num = id_of_file_slice(abs_dcm_file)
    if slice_num > slice_max or slice_num < slice_min:
        os.unlink(abs_dcm_file)
        print("Unlinked {}".format(abs_dcm_file))


def id_of_patient_dir(patient_dir):
    id_as_match = re.search("[0-9]+$", patient_dir)
    return int(id_as_match.group(0))


abs_proj_path = 'C:/Users/justin/PycharmProjects/cpsc464-group2'
data_path = 'image_data/manifest-1654812109500/Duke-Breast-Cancer-MRI'
abs_data_path = os.path.join(abs_proj_path, data_path)
dcm_files = dcm_utils.dcm_dir_list(abs_data_path)
n_dcm_files = len(dcm_files)

dcm_files_data = []
for file in dcm_files:
    dcm_files_data.append(file.split('\\'))

start_slices_path = 'resources/start_slices.txt'
end_slices_path = 'resources/end_slices.txt'
abs_start_slices_path = '{}/{}'.format(abs_proj_path, start_slices_path)
abs_end_slices_path = '{}/{}'.format(abs_proj_path, end_slices_path)
start_slices = np.genfromtxt(abs_start_slices_path, dtype=int)
end_slices = np.genfromtxt(abs_end_slices_path, dtype=int)

input("Type any non-empty string and press enter to proceed with slice filtering ...")

for i in range(n_dcm_files):
    patient_id = id_of_patient_dir(dcm_files_data[i][0])
    start_slice = start_slices[patient_id - 1]
    end_slice = end_slices[patient_id - 1]
    abs_dcm_file = os.path.join(data_path, dcm_files[i])
    filter_file_slice(abs_dcm_file, start_slice, end_slice)
