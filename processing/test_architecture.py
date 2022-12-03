import torch
import numpy as np

import sys
sys.path.append('C:/Users/justin/PycharmProjects/cpsc464-group2/nyu_algos')

from architecture import create_model
from dcm_utils import create_tensor


# creates an empty float tensor from a numpy shape
def create_empty_tensor(shape):
    return create_tensor(np.empty(shape))


examp_model = create_model()
print("Model architecture: ")
print(examp_model)
print("Outcome of applying the untrained model to two 256x256 images:")
bat2 = create_empty_tensor((2, 1, 1, 256, 256))
print(examp_model(bat2))
print("Outcome of applying the untrained model to four 128x128 images:")
bat4 = create_empty_tensor((4, 1, 1, 128, 128))
print(examp_model(bat4))
