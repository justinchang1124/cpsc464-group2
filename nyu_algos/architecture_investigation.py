from networks.models import MRIModels
import torch
import numpy as np


# dummy class
class ParamTest:
    pass


# create a model that can classify a Tensor(batch_size, n_channels, z, x, y) into a stage
def create_model(resnet_groups=16):
    s_params = ParamTest()  # create a class on the fly
    s_params.weights = False  # necessary to run the constructor
    s_params.architecture = '3d_resnet18_fc'  # necessary for classification
    s_params.resnet_groups = resnet_groups  # the only parameter that changes performance
    s_params.input_type = 'birads'  # this doesn't seem to do anything but is good for covering bases
    s_params.network_modification = None  # necessary to run the constructor
    # 5 classes (0, 1, 2, 3, 4)
    return MRIModels(s_params, num_classes=5).model


# creates a float tensor from a numpy array
def create_tensor(np_array):
    return torch.from_numpy(np_array).float()


# creates an empty float tensor from a numpy shape
def create_empty_tensor(shape):
    return create_tensor(np.empty(shape))


examp_model = create_model()
bat1 = create_empty_tensor((1, 1, 1, 256, 256))
bat2 = create_empty_tensor((2, 1, 1, 256, 256))
print("Outcome of applying the model to one 256x256 image:")
print(examp_model(bat1))
print("Outcome of applying the model to two 256x256 images:")
print(examp_model(bat2))