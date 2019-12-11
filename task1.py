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
        #transforms.Resize(256),
        transforms.Resize((224,224)),
        #transforms.CenterCrop(224),
        #transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    
    'test': transforms.Compose([
        transforms.Resize((224,224)),   
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
}

dataset = ROB535Dataset(data_dir='data/rob535-fall-2019-task-1-image-classification', phase='train', transforms=data_transforms['test'])
dataloader = DataLoader(dataset, batch_size = 32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnext50_32x4d(pretrained=False)
num_in_features = 2048

# Freezing parameters
for param in model.parameters():
    param.require_grad = False
  
# Create Custom Classifier
def build_classifier(num_in_features, hidden_layers, num_out_features):
    classifier = torch.nn.Sequential()
    # when we don't have any hidden layers
    if hidden_layers == None:      
        classifier.add_module('fc0', torch.nn.Linear(num_in_features, len(dataset.classes)))    # Change 196 to 4 (4 classes in our dataset)
    #when we have hidden layers
    else:
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        classifier.add_module('fc0', torch.nn.Linear(num_in_features, hidden_layers[0]))
        classifier.add_module('relu0', torch.nn.ReLU())
        classifier.add_module('drop0', torch.nn.Dropout(.6))
        
        for i, (h1, h2) in enumerate(layer_sizes):
            classifier.add_module('fc'+str(i+1), torch.nn.Linear(h1, h2))
            classifier.add_module('relu'+str(i+1), torch.nn.ReLU())
            classifier.add_module('drop'+str(i+1), torch.nn.Dropout(.5))
        classifier.add_module('output', torch.nn.Linear(hidden_layers[-1], num_out_features))
        
    return classifier

#define our hidden layers
hidden_layers = [1000] #None
classifier = build_classifier(num_in_features, hidden_layers, 4)
model.fc = classifier
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)

if device == 'cpu':
    model.load_state_dict(torch.load('models/team10_trained_resnext_epoch5.pt', map_location=torch.device('cpu')))
elif device == 'cuda':
    model.load_state_dict(torch.load('models/team10_trained_resnext_epoch5.pt'))

'''
from torchsummary import summary
if device == 'cpu':
    summary(model.cpu(), (3,224,224))
elif device == 'cuda':
    summary(model.cuda(), (3,224,224))
'''

test_dir = 'data/rob535-fall-2019-task-1-image-classification'
with torch.no_grad():
    print("Predictions on Test Set:")
    model.eval()
    test_dataset = ROB535Dataset(data_dir=test_dir, phase='test', transforms=data_transforms['test'])
    test_dataloader = DataLoader(test_dataset, batch_size = 64, shuffle=False, num_workers=2)

    image_names = [Path(i) for i in test_dataloader.dataset.imgs]
    pred = []

    results = []
    file_names = []
    predicted_car = []
    predicted_class = []
    for inputs in test_dataloader:
        inputs = inputs.to(device)
        #labels = labels.to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs, 1) 

        for i in range(len(inputs)):
            file_names.append(image_names[i])
            #predicted_car.append(int(pred[i] + 1))
            # Is there an off-by-1 error?
            predicted_car.append(int(pred[i]))
results.append((file_names, predicted_car))
df = pd.DataFrame({'guid/image': image_names, 'label': results[0][1]})
df.to_csv('pre_predictions2.csv', index=False)
df['guid/image'] = df['guid/image'].str.replace('data/rob535-fall-2019-task-1-image-classification/data-2019/test/','')
df['guid/image'] = df['guid/image'].str.replace('_image.jpg','')
pd.set_option('display.max_colwidth', -1)
df = df.sort_values(by=['guid/image'])
print(df)
df.to_csv('predictions.csv', index=False)