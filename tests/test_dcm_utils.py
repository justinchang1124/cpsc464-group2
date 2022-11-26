import dcm_utils
import numpy as np
import os
import matplotlib
from matplotlib.animation import FuncAnimation

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
valid1 = np.array([[1, 2, 3], [4, 5, 6]])
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


n_dcm_files = len(dcm_files)

dcm_files_data = []
for file in dcm_files:
    dcm_files_data.append(file.split('\\'))

dcm_studies = [None] * n_dcm_files
for i in range(n_dcm_files):
    dcm_studies[i] = os.path.join(*dcm_files_data[i][:-1])  # splat
dcm_studies = sorted(list(set(dcm_studies)))

input("Press enter to proceed with animating ...")

for dcm_study in dcm_studies:
    abs_dcm_study = os.path.join(abs_data_path, dcm_study)
    dcm_images = dcm_utils.open_dcm_folder(abs_dcm_study)
    dcm_utils.animate_dcm_images(dcm_images, False)


