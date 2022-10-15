import dcm_utils
import os

abs_proj_path = 'C:/Users/justin/PycharmProjects/cpsc464-group2'
data_path = 'image_data/manifest-1654812109500/Duke-Breast-Cancer-MRI'
abs_data_path = os.path.join(abs_proj_path, data_path)
dcm_files = dcm_utils.dcm_dir_list(abs_data_path)
n_dcm_files = len(dcm_files)

dcm_files_data = []
for file in dcm_files:
    dcm_files_data.append(file.split('\\'))

dcm_studies = [None] * n_dcm_files
for i in range(n_dcm_files):
    dcm_studies[i] = os.path.join(*dcm_files_data[i][:-1])  # splat
dcm_studies = sorted(list(set(dcm_studies)))

dcm_studies = dcm_studies[4:]

# test whether every study can be opened and viewed
for dcm_study in dcm_studies:
    print("Viewing study: {}".format(dcm_study))
    abs_dcm_study = os.path.join(abs_data_path, dcm_study)
    dcm_images = dcm_utils.open_dcm_folder(abs_dcm_study)
    # this last step is for frame-limiting images
    dcm_utils.animate_dcm_images(dcm_images, False)

