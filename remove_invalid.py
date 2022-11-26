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
n_dcm_studies = len(dcm_studies)

# test whether every study can be opened and contains an image
invalid_studies = []
for i in range(n_dcm_studies):
    dcm_study = dcm_studies[i]
    print("Study {}/{}: {}".format(i, n_dcm_studies, dcm_study))
    abs_dcm_study = os.path.join(abs_data_path, dcm_study)

    folder_abs_dcm_files = dcm_utils.dcm_dir_list(abs_dcm_study, ret_abs=True)
    if len(folder_abs_dcm_files) > 0:
        try:
            dcm_utils.open_dcm_with_image(folder_abs_dcm_files[0])
        except (Exception,):
            # if that folder contains an invalid study
            invalid_studies.append(dcm_study)


valid_studies = [x for x in dcm_studies if x not in invalid_studies]

print("(Valid {}) + (Invalid {}) = (Total {})".format(
    len(valid_studies), len(invalid_studies), len(dcm_studies)
))

print("*** Invalid Studies ***")
print(invalid_studies)

input("Type any non-empty string and press enter to proceed with eliminating invalid studies ...")

for dcm_study in invalid_studies:
    abs_dcm_study = os.path.join(abs_data_path, dcm_study)
    folder_abs_dcm_files = dcm_utils.dcm_dir_list(abs_dcm_study, ret_abs=True)
    for abs_dcm_file in folder_abs_dcm_files:
        os.unlink(abs_dcm_file)
        print("Unlinked {}".format(abs_dcm_file))
