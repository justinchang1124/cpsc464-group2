import os
import pydicom as dicom
import matplotlib
import glob
from matplotlib.animation import FuncAnimation
import cv2

matplotlib.use("TkAgg") # Qt5Agg is terrible
import matplotlib.pyplot as plt


# recursively gets the locations of all DCM files in abs_base_path/folder
# abs_dir_path: an absolute path to a directory
# abs: if True, return absolute locations (otherwise return relative locations)
def dcm_dir_list(abs_dir_path, abs=False):
    if not (os.path.isabs(abs_dir_path) and os.path.isdir(abs_dir_path)):
        raise ValueError("Not an absolute value path!")
    cur_cwd = os.getcwd()
    os.chdir(abs_dir_path)
    result = list(glob.iglob('**/*.dcm', recursive=True))
    os.chdir(cur_cwd)
    if abs:
        n = len(result)
        for i in range(n):
            result[i] = os.path.join(abs_dir_path, result[i])
    return result


# opens a specific DCM image
def open_dcm_image(abs_dcm_file):
    if not (os.path.isabs(abs_dcm_file) and os.path.isfile(abs_dcm_file)):
        raise ValueError("Not an absolute file path!")
    return dicom.dcmread(abs_dcm_file).pixel_array


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
    dcm_filenames = dcm_dir_list(abs_dir_path)
    n_dcm_files = len(dcm_filenames)
    result = [None] * n_dcm_files
    for i in range(n_dcm_files):
        abs_dcm_file = os.path.join(abs_dir_path, dcm_filenames[i])
        result[i] = open_dcm_image(abs_dcm_file)
    return result


# animates a list of DCM images [each a 2D numpy array]
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

    # 17 milliseconds per frame yields 58-60 FPS
    fig.canvas.mpl_connect('button_press_event', on_click)
    anim = FuncAnimation(plt.gcf(), update, frames=n, interval=17, blit=True)
    return anim
