import time
import json
import copy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from torchvision import datasets, transforms, models
from torch import nn,optim
import torch.nn.functional as F
from torch.utils.data import *
from PIL import Image
from collections import OrderedDict
from pathlib import Path

# check if GPU is available
train_on_gpu = torch.cuda.is_available()

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
## If training phase ...

model_name = 'resnext'
model = models.resnext50_32x4d(pretrained=True)
num_in_features = 2048

# Freezing parameters
for param in model.parameters():
    param.require_grad = False

hidden_layers = [1000]
new_classifier = nn.Sequential()
new_classifier.add_module('fc0', nn.Linear(num_in_features, hidden_layers[0]))
new_classifier.add_module('relu0', nn.ReLU())
new_classifier.add_module('drop0', nn.Dropout(.6))
new_classifier.add_module('output', nn.Linear(hidden_layers[0], 4))

 # Defining model hyperparameters
model.fc = new_classifier
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)

# Loading pretrained weights
if not train_on_gpu:
    model.load_state_dict(torch.load('models/team10_trained_resnext_epoch5.pt', map_location=torch.device('cpu')))
else:
    model.load_state_dict(torch.load('models/team10_trained_resnext_epoch5.pt'))
model.to(device)


test_dir = 'data/rob535-fall-2019-task-1-image-classification/data-2019/test/'

with torch.no_grad():
    print("Predictions on Test Set:")
    model.eval()  
    dataset = datasets.ImageFolder(test_dir,transform=data_transforms['test'])
    testloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    image_names = []
    pred = []
    for index in testloader.dataset.imgs:
        tmp = index[0].replace('data/rob535-fall-2019-task-1-image-classification/data-2019/test/','')
        tmp = tmp.replace('_image.jpg', '')
        image_names.append(Path(tmp))
    
    
    results = []
    file_names = []
    predicted_car = []
    predicted_class = []

    for inputs,labels in testloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1) 
    
        for i in range(len(inputs)):
            file_names.append(image_names[i])
            predicted_car.append(int(pred[i]))
      
results.append((file_names, predicted_car))

print("Predictions on Test Set:")
df = pd.DataFrame({'guid/image': image_names, 'label': results[0][1]})
pd.set_option('display.max_colwidth', -1)
df = df.sort_values(by=['guid/image'])
df

df.to_csv('predictions_new2.csv', index=False)