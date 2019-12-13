import time
import json
import copy
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils, models
from PIL import Image
from pathlib import Path
from collections import OrderedDict

from utils.utils import rot, get_bbox, imshow
from data_loaders import ROB535Dataset

# check if GPU is available
train_on_gpu = torch.cuda.is_available()

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.Resize((300,600)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize((300,600)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    'test': transforms.Compose([
        transforms.Resize((300,600)),   
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
}

# phase='train' in order to first load the official model and then make adjustments to it
dataset = ROB535Dataset(data_dir='data/rob535-fall-2019-task-1-image-classification', phase='train', transforms=data_transforms['test'])
dataloader = DataLoader(dataset, batch_size = 32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnext50_32x4d(pretrained=True)
num_in_features = 2048

# Freezing parameters
for param in model.parameters():
    param.require_grad = False
  
# Create Custom Classifier
hidden_layers = [1000]
new_classifier = torch.nn.Sequential()
new_classifier.add_module('fc0', torch.nn.Linear(num_in_features, hidden_layers[0]))
new_classifier.add_module('relu0', torch.nn.ReLU())
new_classifier.add_module('drop0', torch.nn.Dropout(.6))
new_classifier.add_module('output', torch.nn.Linear(hidden_layers[0], 4))

 # Defining model hyperparameters
model.fc = new_classifier
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)

if not train_on_gpu:
    model.load_state_dict(torch.load('models/team10_trained_resnext50_final.pt', map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load('models/team10_trained_resnext50_final.pt'))
model.to(device)

from torchsummary import summary
if device == 'cpu':
    summary(model.cpu(), (3,224,224))
elif device == 'cuda':
    summary(model.cuda(), (3,224,224))

test_dir = 'data/rob535-fall-2019-task-1-image-classification'
with torch.no_grad():
    print("Evaluating test data: ")
    model.eval()
    test_dataset = ROB535Dataset(data_dir=test_dir, phase='test', transforms=data_transforms['test'])
    test_dataloader = DataLoader(test_dataset, batch_size = 64, shuffle=False, num_workers=2)

    image_names = []
    pred = []
    for index in test_dataloader.dataset.imgs:
        tmp = index.replace('data/rob535-fall-2019-task-1-image-classification/data-2019/test/','')
        tmp = tmp.replace('_image.jpg', '')
        image_names.append(Path(tmp))

    results = []
    file_names = []
    predicted_car = []
    predicted_class = []

    for inputs in test_dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1)

        for i in range(len(inputs)):
            file_names.append(image_names[i])
            predicted_car.append(int(pred[i]))
results.append((file_names, predicted_car))

# Create new dataframe
df = pd.DataFrame({'guid/image': image_names, 'label': results[0][1]})
pd.set_option('display.max_colwidth', -1)
df = df.sort_values(by=['guid/image'])
print(df)

# Write the final submission csv file
df.to_csv('predictions.csv', index=False)