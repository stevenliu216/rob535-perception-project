import time
import json
import copy
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from PIL import Image
from pathlib import Path
from collections import OrderedDict

from utils.utils import rot, get_bbox, imshow
from data_loaders import ROB535Dataset

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    'test': transforms.Compose([
        transforms.Resize((224,224)),   
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
}


test_dir = 'data/rob535-fall-2019-task-1-image-classification'
test_dataset = ROB535Dataset(data_dir=test_dir, phase='test', transforms=data_transforms['test'])
test_dataloader = DataLoader(test_dataset, batch_size = 64, shuffle=False, num_workers=2)
images = next(iter(test_dataloader))
class_names = ['None', 'Cars', 'Commercial', 'Other']
fig, axes = plt.subplots(figsize=(16,5), ncols=6)
for ii in range(6):
    ax = axes[ii]
    ax.set_title(class_names[0])
    imshow(images[ii], ax=ax, normalize=True)
plt.show()
