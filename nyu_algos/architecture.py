from networks.models import MRIModels


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



