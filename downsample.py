import dcm_utils
import numpy as np
import os

abs_proj_path = 'C:/Users/justin/PycharmProjects/cpsc464-group2'
data_path = 'image_data/manifest-1654812109500/Duke-Breast-Cancer-MRI'
abs_data_path = os.path.join(abs_proj_path, data_path)
abs_dcm_files = dcm_utils.dcm_dir_list(abs_data_path, ret_abs=True)
n_dcm_files = len(abs_dcm_files)

min_x = 1000000000  # 1 billion
min_y = 1000000000  # 1 billion

for i in range(n_dcm_files):
    dcm_data = dcm_utils.open_dcm_with_image(abs_dcm_files[i])
    dcm_shape = dcm_data.pixel_array.shape
    min_x = min(min_x, dcm_shape[0])
    min_y = min(min_y, dcm_shape[1])
    if i % 100 == 0:
        print("{}/{}: ({}, {})".format(i, n_dcm_files, min_x, min_y))

print("Final resolution: {}, {}".format(min_x, min_y))
input("Type any non-empty string and press enter to proceed with downscaling ...")

i_shape = (min_x, min_y)  # intended shape
i_shape = (512, 512)

# test: attempting to downscale and reopen
test_n = 10

for abs_dcm_file in abs_dcm_files[:test_n]:
    dcm_utils.downscale_dcm(abs_dcm_file, i_shape)

for abs_dcm_file in abs_dcm_files[:test_n]:
    test_image = dcm_utils.open_dcm_image(abs_dcm_file)
    print(test_image.shape)
    print(np.mean(test_image))
    print(np.std(test_image))

# proceed with full downscaling
for abs_dcm_file in abs_dcm_files:
    dcm_utils.downscale_dcm(abs_dcm_file, i_shape)
