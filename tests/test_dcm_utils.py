import image_testing
import os

abs_proj_path = 'C:/Users/justin/PycharmProjects/cpsc464-group2'
data_path = 'image_data/manifest-1654812109500/Duke-Breast-Cancer-MRI'
abs_data_path = os.path.join(abs_proj_path, data_path)
dcm_files = image_testing.dcm_dir_list(abs_data_path)
n_dcm_files = len(dcm_files)

dcm_files_data = []
for file in dcm_files:
    dcm_files_data.append(file.split('\\'))

dcm_studies = [None] * n_dcm_files
for i in range(n_dcm_files):
    dcm_studies[i] = os.path.join(*dcm_files_data[i][:-1])  # splat
dcm_studies = sorted(list(set(dcm_studies)))

# for dcm_study in dcm_studies:
#     print(dcm_study)

for i in range(6):
    dcm_study_examp = dcm_studies[i]
    dcm_images_examp = image_testing.open_dcm_folder(os.path.join(abs_data_path, dcm_study_examp))
    image_testing.animate_dcm_images(dcm_images_examp)
