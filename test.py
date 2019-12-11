import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

from utils.utils import rot, get_bbox, imshow

from data.data_loaders import ROB535Dataset

txfs = transforms.Compose([transforms.RandomRotation(30),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test = ROB535Dataset(labels_csv_file='data/trainval/labels.csv', data_dir='data/trainval/', transforms=txfs)
dataloader = DataLoader(test, batch_size = 32, shuffle = True)

