import os
import pandas as pd
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

class ROB535Dataset(Dataset):
    '''
    Dataset provided from ROB535 Kaggle.
    
    Following command downloads it:
        kaggle competitions download -c rob535-fall-2019-task-1-image-classification
    '''

    def __init__(self, data_dir='data/rob535-fall-2019-task-1-image-classification', phase='test', transforms=None):
        '''
        Args:
            labels_csv_file (string): Path to labels.csv
            data_dir (string): Path to directory containing trainval, test folders
            transforms: Data transforms to apply
        Returns:
            tuple: (image_sample, label) where image_sample is an image and label is the class label
        '''
        self.phase = phase
        if self.phase == 'train':
            self.df = pd.read_csv(os.path.join(data_dir, 'data-2019', 'trainval', 'labels.csv'))
        elif self.phase == 'test':
            self.df = pd.DataFrame({'guid/image': glob.glob(data_dir+'/data-2019/test/*/*_image.jpg')})
        
        self.data_dir = data_dir
        self.transforms = transforms
        self.imgs = self.df['guid/image'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            idx = idx.tolist()

        if self.phase == 'train':
            image_name = os.path.join(self.data_dir,'data-2019','trainval',self.df.iloc[index, 0] + '_image.jpg')
            image = Image.open(image_name)
            if self.transforms is not None:
                image = self.transforms(image)
            label = self.df.iloc[index, 1]
            return image, label
        elif self.phase == 'test':
            image_name = self.df.iloc[index, 0]
            image = Image.open(image_name)
            if self.transforms is not None:
                image = self.transforms(image)
            return image


class ROB535Dataset_with_bbox(Dataset):
    '''
    Dataset provided from ROB535 Kaggle.
    
    Following command downloads it:
        kaggle competitions download -c rob535-fall-2019-task-1-image-classification
    '''

    def __init__(self, data_dir='data/rob535-fall-2019-task-1-image-classification', phase='test', transforms=None):
        '''
        Args:
            labels_csv_file (string): Path to labels.csv
            data_dir (string): Path to directory containing trainval, test folders
            transforms: Data transforms to apply
        Returns:
            tuple: (image_sample, label) where image_sample is an image and label is the class label
        '''
        self.phase = phase
        if self.phase == 'train':
            self.df = pd.read_csv(os.path.join(data_dir, 'data-2019', 'trainval', 'labels.csv'))
        elif self.phase == 'test':
            self.df = pd.DataFrame({'guid/image': glob.glob(data_dir+'/data-2019/test/*/*_image.jpg')})
        
        self.data_dir = data_dir
        self.transforms = transforms
        self.imgs = self.df['guid/image'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            idx = idx.tolist()

        if self.phase == 'train':
            image_name = os.path.join(self.data_dir,'data-2019','trainval',self.df.iloc[index, 0] + '_image.jpg')
            proj_name = os.path.join(self.data_dir,'data-2019','trainval',self.df.iloc[index, 0] + '_proj.bin')
            bbox_name = os.path.join(self.data_dir,'data-2019','trainval',self.df.iloc[index, 0] + 'bbox.bin')

            image = Image.open(image_name)
            proj = np.fromfile(proj_name, dtype=np.float32)
            proj.resize([3,4])
            proj = torch.from_numpy(proj)
            bbox = np.fromfile(bbox_name, dtype=np.float32)
            bbox = bbox.reshape([-1, 11])
            bbox = torch.from_numpy(bbox)

            if self.transforms is not None:
                image = self.transforms(image)
            label = self.df.iloc[index, 1]
            return image, label
        elif self.phase == 'test':
            image_name = self.df.iloc[index, 0]
            image = Image.open(image_name)
            if self.transforms is not None:
                image = self.transforms(image)
            return image