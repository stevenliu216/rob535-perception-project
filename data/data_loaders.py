import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

class ROB535Dataset(Dataset):
    '''
    Dataset provided from ROB535 Kaggle.
    
    Following command downloads it:
        kaggle competitions download -c rob535-fall-2019-task-1-image-classification
    '''

    def __init__(self, labels_csv_file='data/trainval/labels.csv', data_dir='data/trainval/', transforms=None):
        '''
        Args:
            labels_csv_file (string): Path to labels.csv
            data_dir (string): Path to directory containing trainval, test folders
            transforms: Data transforms to apply
        Returns:
            tuple: (image_sample, label) where image_sample is an image and label is the class label
        '''
        self.df = pd.read_csv(labels_csv_file)
        self.data_dir = data_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            idx = idx.tolist()

        image_name = self.data_dir + self.df.iloc[index, 0] + '_image.jpg'
        image = Image.open(image_name)
        if self.transforms is not None:
            image = self.transforms(image)
        label = self.df.iloc[index, 1]
        return image, label
