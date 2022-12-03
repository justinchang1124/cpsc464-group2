import os
import pydicom as dicom
import glob
import cv2
import numpy as np
import torch
import torchio


def validate_abs_dir_path(abs_dir_path):
    """
    Raises an error if abs_dir_path is not absolute or not a directory.

    :param abs_dir_path: expected to be an absolute path to a directory that exists
    :return: void
    """
    if not (os.path.isabs(abs_dir_path) and os.path.isdir(abs_dir_path)):
        raise ValueError("Not abs_dir_path (absolute path to a directory that exists)!")


def prepend_abs(abs_dir_path, rel_paths):
    """
    Prepends an absolute dir path to a list of relative paths.

    :param abs_dir_path: see validate_abs_dir_path(abs_dir_path)
    :param rel_paths: expected to be a tuple / list of relative paths
    :return: a list of the form abs_dir_path/rel_paths
    """
    validate_abs_dir_path(abs_dir_path)
    return [os.path.join(abs_dir_path, rel_path) for rel_path in rel_paths]


def dcm_dir_list(abs_dir_path, ret_abs=False):
    """
    Lists all DCM files within an absolute dir path.

    :param abs_dir_path: see validate_abs_dir_path(abs_dir_path)
    :param ret_abs: assumed to be True (return absolute paths) or False (return relative paths)
    :return: a list of the form abs_dir_path/*
    """
    validate_abs_dir_path(abs_dir_path)
    cur_cwd = os.getcwd()
    os.chdir(abs_dir_path)
    rel_paths = list(glob.iglob('**/*.dcm', recursive=True))
    os.chdir(cur_cwd)
    if ret_abs:
        return prepend_abs(abs_dir_path, rel_paths)
    return rel_paths


def validate_norm_ndarray(nnd_cand, n):
    """
    Raises an error if nnd_cand is not an n-dimensional non-jagged numpy array.

    :param nnd_cand: expected to be an n-dimensional non-jagged numpy array
    :param n: assumed to be an integer, expected to be positive
    :return: void
    """
    if n < 1:
        raise ValueError("Validating nnd_cand requires n to be positive!")
    if not isinstance(nnd_cand, np.ndarray):
        raise ValueError("Expected nnd_cand to be a numpy array!")
    if nnd_cand.dtype is np.dtype('object'):
        raise ValueError("Expected nnd_cand to be non-jagged!")
    if len(nnd_cand.shape) != n:
        raise ValueError("Expected [len(nnd_cand.shape) == n] but received [{} != {}]!".format(
            len(nnd_cand.shape), n
        ))


def validate_le(a, b):
    """
    Raises an error if a > b.

    :param a: assumed to be a number
    :param b: assumed to be a number
    :return: void
    """
    if a > b:
        raise ValueError("Expected a <= b but received {} > {}".format(a, b))


def clamp_dcm_image(dcm_image, v_min, v_max):
    """
    Clamps all values in dcm_image to the range [v_min, v_max].
    Note: we assume [v_min, v_max] does not contain [min(dcm_image), max(dcm_image)]

    :param dcm_image: see validate_norm_ndarray(dcm_image, 2)
    :param v_min: assumed to be a real number
    :param v_max: assumed to be a real number
    :return: a two-dimensional non-jagged numpy array
    """
    validate_norm_ndarray(dcm_image, 2)
    validate_le(v_min, v_max)
    return np.clip(dcm_image, v_min, v_max)


def perc_clamp_dcm_image(dcm_image, p_min, p_max):
    """
    Performs percentile-based clamping.
    Note: we assume [p_min, p_max] does not contain [0, 100].

    :param dcm_image: see validate_norm_ndarray(dcm_image, 2)
    :param p_min: assumed to be a real number representing a percentile
    :param p_max: assumed to be a real number representing a percentile
    :return: a two-dimensional non-jagged numpy array
    """
    validate_norm_ndarray(dcm_image, 2)
    validate_le(p_min, p_max)
    flat_img = dcm_image.flatten()
    v_min = np.percentile(flat_img, p_min)
    v_max = np.percentile(flat_img, p_max)
    return np.clip(dcm_image, v_min, v_max)


def downscale_dcm_image(dcm_image, shape):
    """
    Downscales a DCM image to the provided shape.

    :param dcm_image: see validate_norm_ndarray(dcm_image, 2)
    :param shape: assumed to be a tuple of two integers
    :return: a two-dimensional non-jagged numpy array
    """
    validate_norm_ndarray(dcm_image, 2)
    # the final shape must be no greater than the initial shape
    validate_le(shape[0], dcm_image.shape[0])
    validate_le(shape[1], dcm_image.shape[1])
    # the documentation recommends cv2.INTER_AREA for downscaling
    # this also obviates the need to rescale to [0, 255] if the input is [0, 255]
    # see also: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
    return cv2.resize(dcm_image, dsize=shape, interpolation=cv2.INTER_AREA)


def normalize_dcm_image(dcm_image):
    """
    Normalizes the values of an image to [0,1].

    :param dcm_image: see validate_norm_ndarray(dcm_image, 2)
    :return: a two-dimensional non-jagged numpy array
    """
    validate_norm_ndarray(dcm_image, 2)
    img_min = np.min(dcm_image)
    img_max = np.max(dcm_image)
    return (dcm_image - img_min) / (img_max - img_min)


def standardize_dcm_image(dcm_image):
    """
    Standardizes the mean of an image to 0 and the variance to 1.

    :param dcm_image: see validate_norm_ndarray(dcm_image, 2)
    :return: a two-dimensional non-jagged numpy array
    """
    validate_norm_ndarray(dcm_image, 2)
    img_mean = np.mean(dcm_image)
    img_var = np.std(dcm_image)
    return (dcm_image - img_mean) / img_var


def validate_abs_dcm_file(abs_dcm_file):
    """
    Raises an error if abs_dcm_file is not absolute or not a file.

    :param abs_dcm_file: Expected to be an absolute path to a file that exists.
    :return: void
    """
    if not (os.path.isabs(abs_dcm_file) and os.path.isfile(abs_dcm_file)):
        raise ValueError("Not an absolute path to a file!")


def open_dcm_with_image(abs_dcm_file):
    """
    Reads DICOM data from the absolute file path provided, raising an error
    if the file does not contain a valid DICOM image.

    :param abs_dcm_file: see validate_abs_dcm_file(abs_dcm_file)
    :return: a pydicom.dataset.FileDataset
    """
    validate_abs_dcm_file(abs_dcm_file)
    dcm_data = dicom.dcmread(abs_dcm_file)
    validate_norm_ndarray(dcm_data.pixel_array, 2)
    return dcm_data


def downscale_dcm(abs_dcm_file, shape):
    """
    Opens a DICOM file, downscales the pixel array, and saves it.

    :param abs_dcm_file: see validate_abs_dcm_file(abs_dcm_file)
    :param shape: assumed to be a tuple of two integers
    :return: whether downscaling occurred
    """
    dcm_data = open_dcm_with_image(abs_dcm_file)
    px_array = dcm_data.pixel_array
    # no point in resizing if the shape is already correct
    if shape == px_array.shape:
        return False
    downscaled_img = downscale_dcm_image(px_array, shape)
    dcm_data.PixelData = downscaled_img.tobytes()
    dcm_data.Rows, dcm_data.Columns = shape
    dicom.dcmwrite(abs_dcm_file, dcm_data)
    return True


def create_tensor(np_array):
    """
    Creates a Torch tensor from a numpy array.

    :param np_array: a numpy array
    :return: a Torch tensor
    """
    return torch.from_numpy(np_array).float()


def dcm_image_to_tensor4d(dcm_image):
    """
    Converts a dcm_image into a 4D Torch tensor.

    :param dcm_image: see validate_norm_ndarray(dcm_image, 2)
    :return: a four-dimensional Torch tensor (channels = 1, z = 1, x, y)
    """
    validate_norm_ndarray(dcm_image, 2)
    result = np.empty((1, *dcm_image.shape, 1))
    result[0, :, :, 0] = dcm_image
    return create_tensor(dcm_image)


def tensor4d_to_dcm_image(tensor4d):
    """
    Converts a 4D Torch tensor into a dcm_image.

    :param tensor4d: a four-dimensional Torch tensor (channels = 1, x, y, z = 1)
    :return: see validate_norm_ndarray(dcm_image, 2)
    """
    if not torch.is_tensor(tensor4d):
        raise ValueError("Expected a Torch tensor!")
    e_size = tensor4d.size()
    if e_size[0] != 1 or e_size[3] != 1:
        raise ValueError("Expected 4D tensor to have channels = 1, z = 1!")
    result = tensor4d.numpy()
    validate_norm_ndarray(result, 4)
    return result[0, :, :, 0]


augmenter = torchio.transforms.Compose([
    torchio.transforms.RandomFlip(axes=0, flip_probability=0.5),
    torchio.transforms.RandomFlip(axes=1, flip_probability=0.5),
    torchio.transforms.RandomFlip(axes=2, flip_probability=0.5),
    torchio.transforms.RandomAffine(
        scales=0.10,
        degrees=10,
        translation=0,
        p=0.65
    )
])


def augment_tensor4d(tensor4d):
    return augmenter(tensor4d)


print("IMPORTED: dcm_utils")
