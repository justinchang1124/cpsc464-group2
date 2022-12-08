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
    :return: a four-dimensional Torch tensor (channels = 1, x, y, z = 1)
    """
    validate_norm_ndarray(dcm_image, 2)
    result = np.empty((1, *dcm_image.shape, 1))
    result[0, :, :, 0] = dcm_image
    return create_tensor(result)


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
    # torchio.transforms.RandomFlip(axes=2, flip_probability=0.5),  # not needed
    torchio.transforms.RandomAffine(
        scales=0.10,
        degrees=10,
        translation=0,
        p=0.65
    )
])


def augment_tensor4d(tensor4d):
    """
    Augments an image with random flip / affline transformations

    :param tensor4d: a four-dimensional Torch tensor (channels = 1, x, y, z = 1)
    :return: a four-dimensional Torch tensor (channels = 1, x, y, z = 1)
    """
    return augmenter(tensor4d)


def label_to_one_hot(i):
    """
    Converts a label to a one-hot Torch tensor encoding.

    :param i: assumed to be in [0, 1, 2, 3, 4]
    :return: a one-hot Torch tensor encoding
    """
    one_hot = [0, 0, 0, 0, 0]
    one_hot[i] = 1
    return torch.tensor(one_hot).float()


def one_hot_to_label(one_hot):
    """
    Converts a one-hot Torch tensor encoding to a label.

    :param one_hot: a one-hot Torch tensor encoding
    :return: a member of [0, 1, 2, 3, 4]
    """
    label_ind = 0
    for i in range(1, 5):
        if one_hot[i] > one_hot[label_ind]:
            label_ind = i
    return label_ind


def get_argmax_batch(output):
    """
    Converts a batch of one-hot encodings to a list of labels.

    :param output: a batch of one-hot encodings
    :return: a list of labels
    """
    bat_size = output.size()[0]
    result = [None] * bat_size
    for i in range(bat_size):
        result[i] = one_hot_to_label(output[i])
    return result


def diff_lists(list_a, list_b):
    """
    Returns pairwise difference between list_a, list_b.

    :param list_a: a list of numbers
    :param list_b: a list of numbers
    :return: pairwise difference between list_a, list_b
    """
    return [a_i - b_i for a_i, b_i in zip(list_a, list_b)]


def key_get_list(examp_dict, examp_key):
    """
    dict.get() but return [] if not found

    :param examp_dict: a dictionary
    :param examp_key: a key for examp_dict
    :return: examp_dict.get(examp_key), return [] if not found
    """
    result = examp_dict.get(examp_key)
    if result is None:
        return []
    return result


def separate_by_label(values, labels):
    """
    Separates values by labels.

    :param values: a list / tuple of values
    :param labels: a list / tuple of labels
    :return: dict results where results[label] is all values with that label
    """
    n = len(values)
    if n != len(labels):
        raise ValueError("Length of values does not equal length of labels!")
    result = {}

    for i in range(n):
        label = labels[i]
        label_list = key_get_list(result, label)
        label_list.append(values[i])
        result[label] = label_list

    return result


def summarize_group(a_list, r_list):
    """
    Prints a summary for an AI list vs a real list.

    :param a_list: a list of AI (predicted) groups
    :param r_list: a list of real (true) groups
    :return: void
    """
    d_list = diff_lists(a_list, r_list)
    n_sup = 0
    n_sub = 0
    n_cor = 0
    for diff in d_list:
        if diff == 0:
            n_cor += 1
        elif diff > 0:
            n_sup += 1
        else:
            n_sub += 1

    total = len(d_list)
    print("\tAccuracy Rate: {}+, {}-, total {}/{} = {}".format(n_sup, n_sub, n_cor, total, n_cor / total))
    avg_re = sum(r_list) / total
    avg_ai = sum(a_list) / total
    print("\tMean True vs AI: {} vs {}".format(avg_re, avg_ai))
    print("\tVAD = {}".format(np.var(d_list)))
    print("\tMAD = {}".format(sum(map(abs, d_list)) / total))


def summarize_ar_dict(a_dict, r_dict):
    """
    Prints a summary for AI vs real groups.

    :param a_dict: a dictionary of AI (predicted) groups
    :param r_dict: a dictionary of real (true) groups
    :return: void
    """
    grp_keys = a_dict.keys()
    if grp_keys != r_dict.keys():
        raise ValueError("Different keys across dicts!")

    print("--- START SUMMARY ---")
    for grp_key in sorted(grp_keys):
        print("GROUP: {}".format(grp_key))
        summarize_group(a_dict[grp_key], r_dict[grp_key])
    print("---- END SUMMARY ----")


def write_labels(labels, filename):
    """
    Writes labels as a comma-separated string.

    :param labels: a list
    :param filename: the file to which this will be written
    :return: void
    """
    text_file = open(filename, "w")
    text_file.write(','.join([str(lab) for lab in labels]))
    text_file.close()


def read_labels(filename, to_int=False):
    """
    Opens a file and reads all labels inside.

    :param filename: the file from which labels will be read
    :param to_int: assumed to be a bool, whether we cast to int
    :return: a list of ints
    """
    text_file = open(filename, "r")
    labels = []
    for line in text_file:
        labels += line.split(',')
    if to_int:
        for i in range(len(labels)):
            labels[i] = int(labels[i])
    text_file.close()
    return labels



