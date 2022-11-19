import dcm_utils
import numpy as np
import os

abs_proj_path = 'C:/Users/justin/PycharmProjects/cpsc464-group2'
data_path = 'image_data/manifest-1654812109500/Duke-Breast-Cancer-MRI'
abs_data_path = os.path.join(abs_proj_path, data_path)
dcm_files = dcm_utils.dcm_dir_list(abs_data_path)

i_shape = (128, 128)

# attempt downsampling
for dcm_file in dcm_files[0:10]:
    abs_dcm_file = os.path.join(abs_data_path, dcm_file)
    dcm_utils.downsample_dcm_file(abs_dcm_file, i_shape)

# can we still open it?
for dcm_file in dcm_files[0:10]:
    abs_dcm_file = os.path.join(abs_data_path, dcm_file)
    test_image = dcm_utils.open_dcm_image(abs_dcm_file)
    print(test_image.shape)
    print(np.mean(test_image))
    print(np.std(test_image))

apply_all = False
if apply_all:
    for dcm_file in dcm_files:
        abs_dcm_file = os.path.join(abs_data_path, dcm_file)
        dcm_utils.downsample_dcm_file(abs_dcm_file, i_shape)
