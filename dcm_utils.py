import os
import pydicom as dicom
import matplotlib
import glob
from matplotlib.animation import FuncAnimation
import cv2
import numpy as np

matplotlib.use("TkAgg") # Qt5Agg is terrible
import matplotlib.pyplot as plt


# prepends an absolute path to a list of relative paths
def prepend_abs(abs_dir_path, rel_paths):
    if not (os.path.isabs(abs_dir_path) and os.path.isdir(abs_dir_path)):
        raise ValueError("Not an absolute value path!")
    n = len(rel_paths)
    result = [None] * n
    for i in range(n):
        result[i] = os.path.join(abs_dir_path, rel_paths[i])
    return result


# recursively gets the locations of all DCM files in abs_base_path/folder
# abs_dir_path: an absolute path to a directory
# abs: if True, return absolute locations (otherwise return relative locations)
def dcm_dir_list(abs_dir_path, use_abs=False):
    if not (os.path.isabs(abs_dir_path) and os.path.isdir(abs_dir_path)):
        raise ValueError("Not an absolute value path!")
    cur_cwd = os.getcwd()
    os.chdir(abs_dir_path)
    rel_paths = list(glob.iglob('**/*.dcm', recursive=True))
    os.chdir(cur_cwd)
    if use_abs:
        return prepend_abs(abs_dir_path, rel_paths)
    return rel_paths


# opens a specific DCM image
def open_dcm_image(abs_dcm_file, percentile=99):
    if not (os.path.isabs(abs_dcm_file) and os.path.isfile(abs_dcm_file)):
        raise ValueError("Not an absolute file path!")
    px_array = dicom.dcmread(abs_dcm_file).pixel_array
    if len(px_array.shape) != 2:
        raise ValueError("Not a two-dimensional image!")
    cutoff = np.percentile(px_array, percentile)
    px_array[px_array > cutoff] = cutoff
    return px_array / cutoff * 255.0


# opens a set of specific DCM images, allows 3D as well
# def open_dcm_images(abs_dcm_files):
#     result = []
#     for abs_dcm_file in abs_dcm_files:
#         dcm_image = open_dcm_image(abs_dcm_file)
#         n_dim = len(dcm_image.shape)
#         if n_dim == 2:
#             result.append(dcm_image)
#         elif n_dim == 3:
#             for i in range(dcm_image.shape[0]):
#                 result.append(dcm_image[i])
#         else:
#             raise ValueError("Not a 2D or 3D DCM image!")
#
#     return result


# opens a set of specific DCM images
def open_dcm_images(abs_dcm_files):
    n = len(abs_dcm_files)
    result = [None] * n
    for i in range(n):
        result[i] = open_dcm_image(abs_dcm_files[i])
    return result


# resizes a DCM image
def resize_dcm_image(dcm_image, shape):
    return cv2.resize(dcm_image, dsize=shape, interpolation=cv2.INTER_CUBIC)


# opens all DCM images in a folder
def open_dcm_folder(abs_dir_path):
    return open_dcm_images(dcm_dir_list(abs_dir_path, use_abs=True))


# converts a list of DCM images into an array of DCM images, with resize
def dcm_images_to_np3d(dcm_images, shape):
    n = len(dcm_images)
    result = np.empty((n, *shape))  # splat
    for i in range(n):
        result[i] = resize_dcm_image(dcm_images[i], shape)
    return result


# animates a list of DCM images [each a 2D numpy array]
def animate_dcm_images(dcm_images, console_mode=False):
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


print("IMPORTED: dcm_utils")
