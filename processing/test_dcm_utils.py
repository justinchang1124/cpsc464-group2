import dcm_utils
import numpy as np
import os
import matplotlib
from matplotlib.animation import FuncAnimation
import torch

# Qt5Agg does not function
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def animate_dcm_images(dcm_images, console_mode=False):
    """
    Animates a list of DICOM images.

    :param dcm_images: expected to be a list of DICOM images
    :param console_mode: assumed to be True (running from console) or False (running from python3)
    :return: void
    """
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
        dcm_utils.validate_norm_ndarray(dcm_images[i], 2)
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

    # 17 milliseconds per frame yields 58-60 FPS
    fig.canvas.mpl_connect('button_press_event', on_click)

    anim = FuncAnimation(plt.gcf(), update, frames=n, interval=17, blit=True)

    if console_mode:
        plt.show()
        plt.pause(3600 * 24)  # pause for 1 day
        input("To continue, close any open plots and press enter!")
    else:
        print("To continue, close any open plots!")
        plt.show()


def expect_error_validate_norm_ndarray(nnd_cand, n):
    try:
        dcm_utils.validate_norm_ndarray(nnd_cand, n)
    except (Exception,):
        print("validate_norm_ndarray correctly raised an error!")


abs_proj_path = 'C:/Users/justin/PycharmProjects/cpsc464-group2'
data_path = 'image_data/manifest-1654812109500/Duke-Breast-Cancer-MRI'
abs_data_path = os.path.join(abs_proj_path, data_path)

# test validate_abs_dir_path:
dcm_utils.validate_abs_dir_path(abs_proj_path)
dcm_utils.validate_abs_dir_path(abs_data_path)
try:
    dcm_utils.validate_abs_dir_path("this/is/not/a/path")
except (Exception,):
    print("validate_abs_dir_path correctly raised an error!")

# test prepend_abs:
print("Result of prepend_abs('C:/Users/justin', ['A', 'B']):")
print(dcm_utils.prepend_abs('C:/Users/justin', ['A', 'B']))

# test dcm_dir_list:
dcm_files = dcm_utils.dcm_dir_list(abs_data_path)
print("Result of head(dcm_dir_list(abs_data_path)):")
print(dcm_files[:6])
print("Result of head(dcm_dir_list(abs_data_path, ret_abs=True)):")
print(dcm_utils.dcm_dir_list(abs_data_path, ret_abs=True)[:6])

# test validate_norm_ndarray:
invalid1 = "dog"
invalid2 = np.array([[1, 2, 3], [4, 5]], dtype=object)
valid1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype('float32')
expect_error_validate_norm_ndarray(invalid1, 1)
expect_error_validate_norm_ndarray(invalid2, 2)
expect_error_validate_norm_ndarray(valid1, 1)
dcm_utils.validate_norm_ndarray(valid1, 2)

# test validate_le
dcm_utils.validate_le(1, 2)
dcm_utils.validate_le(3, 3)
try:
    dcm_utils.validate_le(100, 1)
except (Exception,):
    print("validate_le correctly raised an error!")

# test clamp_dcm_image
print(dcm_utils.clamp_dcm_image(valid1, 3, 7))

# test perc_clamp_dcm_image
print(dcm_utils.perc_clamp_dcm_image(valid1, 5, 95))

# test downscale_dcm_image
print(dcm_utils.downscale_dcm_image(valid1, (2, 2)))

# test normalize_dcm_image
print(dcm_utils.normalize_dcm_image(valid1))

# test standardize_dcm_image
print(dcm_utils.standardize_dcm_image(valid1))

# test label_to_one_hot
print(dcm_utils.label_to_one_hot(3))

# test get_argmax_batch
examp1 = torch.tensor([[-3.6933, -0.9389,  0.3308, -2.0394, -2.4012],
        [-3.4493, -0.8140, -0.5046, -1.4021, -3.3808]])

examp2 = torch.tensor([[0, 0, 1, 0, 0],
                       [0, 0, 0, 1, 0]])

print(dcm_utils.get_argmax_batch(examp1))
print(dcm_utils.get_argmax_batch(examp2))

# test the summarizing functions
ea_dict, er_dict = dcm_utils.separate_by_group(
    [1, 2, 1, 2, 3, 4],
    [1, 2, 1, 2, 1, 2],
    ['a', 'a', 'b', 'b', 'c', 'c']
)
dcm_utils.summarize_ar_dict(ea_dict, er_dict)

aug_flag = input("Type 'y' to augment, any other non-empty string to skip. Press enter to animate ...")

# test manipulating DICOM files
n_dcm_files = len(dcm_files)

dcm_files_data = []
for file in dcm_files:
    dcm_files_data.append(file.split('\\'))

dcm_studies = [None] * n_dcm_files
for i in range(n_dcm_files):
    dcm_studies[i] = os.path.join(*dcm_files_data[i][:-1])  # splat
dcm_studies = sorted(list(set(dcm_studies)))

for dcm_study in dcm_studies:
    abs_dcm_study = os.path.join(abs_data_path, dcm_study)
    abs_study_files = dcm_utils.dcm_dir_list(abs_dcm_study, ret_abs=True)
    n_study_files = len(abs_study_files)
    dcm_images = [None] * n_study_files
    for i in range(n_study_files):
        dcm_data = dcm_utils.open_dcm_with_image(abs_study_files[i])
        img_clamp = dcm_utils.perc_clamp_dcm_image(dcm_data.pixel_array, 1, 99)
        img_norm = dcm_utils.normalize_dcm_image(img_clamp)
        if aug_flag == "y":
            img_tensor = dcm_utils.dcm_image_to_tensor4d(img_norm)
            img_aug = dcm_utils.augment_tensor4d(img_tensor)
            img_norm = dcm_utils.tensor4d_to_dcm_image(img_aug)
        dcm_images[i] = img_norm * 255.0
    animate_dcm_images(dcm_images, False)


