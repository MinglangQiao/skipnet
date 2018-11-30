import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

from torch import autograd
import numpy as np

MEAN_VALUE = np.array([123.68, 116.779, 103.939])   # RGB, refere to this: https://github.com/tensorflow/models/issues/517
# MEAN_VALUE = np.array([103.939, 116.779, 123.68])   # BGR
MEAN_VALUE = MEAN_VALUE[:,None, None]

c1 = np.array(MEAN_VALUE)

c2 = c1 + MEAN_VALUE
print(MEAN_VALUE)
print('>>> : ', c1)
print('>>> c2: ', c2)