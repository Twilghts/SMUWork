import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

model = resnet18(weights=ResNet18_Weights.DEFAULT)

for name, param in model.named_parameters():
    print(name, param.shape)
