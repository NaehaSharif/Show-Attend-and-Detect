#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 12:15:23 2021

@author: naeha
"""
import numpy as np
import os
import glob
import torch
seed = 3
import torchvision.transforms as transforms
torch.manual_seed(seed)
import json
import torch
import torchvision
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from torch.utils.data import Sampler
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import collections
import time
import tqdm
import copy
from PIL import Image
from skimage.transform import resize
#https://www.kaggle.com/veb101/transfer-learning-using-efficientnet-models
#import seaborn as sns

#https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling
'''
class 0- 2=<x<6
class 1- x<2
class2-  x>=6 '''


def get_idx(dir_name, binary_dir):
    
        fnames_dic = np.load(binary_dir,allow_pickle=True)
        
        if not isinstance(dir_name,str):
            fnames = []
            for curr_dir in dir_name:
                fnames += glob.glob(os.path.join(curr_dir, "*"))
        else:
            fnames = glob.glob(os.path.join(dir_name, "*"))
            
        
        d = dict(enumerate(fnames_dic.flatten(), 1))
        d=d[1]
        check_names = list(d.keys())
        #self.fname_list=[s for s in self.fnames if s.split('/')[-1].split('.')[0]+'.dcm'  in self.check_names]
        fname_list=[s for s in fnames if s.split('/')[-1][:-4]  in check_names]
        values = np.array([d[s.split('/')[-1][:-4]] for s in fname_list])
        labels1= values>=6.0
        labels2=values<2
        
        labels1=labels1*2
        labels2=labels2*1
        labels=values*0
        # labels=values.astype(np.float)
        labels_all=labels1+labels+labels2
        
        data_ = {'image_id': fname_list,
        'labels': values
        }

        df = pd.DataFrame(data_, columns= ['image_id', 'labels'])

        df.to_csv ('id_to_labels.csv', index = False, header=True)

        return fname_list, values
        
def Rescale (sample,output_size):
        #rescaled = resize(sample[0], output_size, mode='constant')
        rescaled=sample[0]
        rescaled= np.repeat(rescaled[:,:, np.newaxis], 3, -1)
        return (rescaled, sample[1])

def ToTensor_(sample):
        image, label = sample
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return (torch.FloatTensor(image), label)
    
    
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)    



class DicomDataset(Dataset):
    
    def __init__(self, label_csv_name, label_fine, level,  transforms=None):
        """
        dir_name= names of dicom directories
        binary_dir: directory where name to label file (.npy) exists
        transforms: ex. ToTensor
        load all file names
        allocate their classes"""
        self.level=level 
        self.df_labels = pd.read_csv(label_csv_name)
        self.df_fine_labels = np.load(label_fine, allow_pickle=True)
        
        self.transforms = transforms
        self.d= self.df_fine_labels.flatten()

        self.d=self.d[0]
        
    def __len__(self):
        return len(self.df_labels)

    def __getitem__(self, idx):
        


        label = np.array(self.df_labels[self.df_labels['image_id'] == idx]['labels'].values[0]/24.0)
        
        #print(idx) # idx is the full path
        
        fine_label = np.array(self.d[idx.split('/')[-1][:-4]])
        fine_label = torch.from_numpy(fine_label.astype(np.float32))
        # this is where we can work on the splitting
        #print('fine_all',fine_label)
        
        
        if self.level=='ant':
           f=[0.,0.,0.,0.,0.,0.]
           f[1:5]=[fine_label[1],fine_label[3],fine_label[5],fine_label[7]]
           fine_label=torch.tensor(f)
           label=torch.sum(fine_label)
          
        elif self.level=='post':
           f=[0.,0.,0.,0.,0.,0.]
           f[1:5]=[fine_label[2],fine_label[4],fine_label[6],fine_label[8]]
           fine_label=torch.tensor(f)
           label=torch.sum(fine_label)
        
#        if self.level=='up':
#          f=torch.tensor([0.,0.,0.,0.]) 
#          f[1]=  torch.sum(fine_label[1:3])
#          f[2]= torch.sum(fine_label[3:5])
#          fine_label=f
#          label=torch.sum(fine_label)
#          #print('sum_up',label)
#        
#        
#         if self.level=='down':
#           f=torch.tensor([0.,0.,0.,0.]) 
#           f[1]=torch.sum(fine_label[5:7])
#           f[2]=torch.sum(fine_label[7:-1])
#           fine_label=f
#           label=torch.sum(fine_label)
#        
#        if self.level=='up_ant':
#          f=torch.tensor([0.,0.,0.,0.])
#          f[1:3]=fine_label[[1,3]]
#          fine_label=f
#          label=torch.sum(fine_label)
#          
#        if self.level=='up_post':
#          f=torch.tensor([0.,0.,0.,0.])
#          f[1:3]=fine_label[[2,4]]
#          fine_label=f
#          label=torch.sum(fine_label)
#        
#        if self.level=='down_ant':
#          f=torch.tensor([0.,0.,0.,0.])
#          f[1:3]=fine_label[[5,7]]
#          fine_label=f
#          label=torch.sum(fine_label)  
#        
#        if self.level=='down_post':
#          f=torch.tensor([0.,0.,0.,0.])
#          f[1:3]=fine_label[[6,8]]
#          fine_label=f
#          label=torch.sum(fine_label)
        
        # if self.level=='upa':
        #   f=torch.tensor([0.,0.,0.,0.])
        #   f[1:3]=fine_label[1:3]
        #   fine_label=f
        #   label=torch.sum(fine_label)
         
        # if self.level=='upb':
        #   f=torch.tensor([0.,0.,0.,0.]) 
        #   f[1:3]=fine_label[3:5]
        #   fine_label=f
        #   label=torch.sum(fine_label)
          
        # if self.level=='downa':
        #   f=torch.tensor([0.,0.,0.,0.])
        #   f[1:3]=fine_label[5:7]
        #   fine_label=f
        #   label=torch.sum(fine_label)
          
        # if self.level=='downb':
        #   f=torch.tensor([0.,0.,0.,0.])
        #   f[1:3]=fine_label[7:-1]
        #   fine_label=f
        #   label=torch.sum(fine_label)
        
         
        
        
        # if self.level=='four':
        #   f=torch.tensor([0.,0.,0.,0.,0., 0.]) 
        #   f[1]=  torch.sum(fine_label[1:3])
        #   f[2]= torch.sum(fine_label[3:5])
        #   f[3]=  torch.sum(fine_label[5:7])
        #   f[4]= torch.sum(fine_label[7:9])
        #   fine_label=f
        #   label=torch.sum(fine_label)
          
         
          
        fine_label=torch.Tensor(fine_label).long()
        #label = torch.from_numpy(label.astype(np.float32))
        
        img = np.load(idx, allow_pickle=True).astype(np.float32)
        
        if self.level=='up' :#or self.level=='up_ant' or self.level=='up_post':
          img=  img[:700,:]
        
        if self.level=='down': #or self.level=='down_ant' or self.level=='down_post':
          img=  img[200:,:]
          
        # if self.level=='upa':
        #   img=  img[:600,:]
        #   #img[:400,:]  
          
        # if self.level=='upb':
        #   img=  img[:600,:]
        #   #img[100:500,:] 
          
       
        
        # if self.level=='downa':
        #   img=  img[300:700,:]
        
        # if self.level=='downb':
        #   img=  img[400:,:]
          
       
        
        sample = (img, label)
        sample=Rescale(sample,(224,224))
        sample= ToTensor_(sample)
        
        img, label=sample
        if self.transforms is not None:
            img = self.transforms(img)
            
           # img = transforms(img)
            
        return (img, label,fine_label, idx)
    
############################################################################################

class ImageSampler(Sampler):
    def __init__(self, 
                 sample_idx,
                 data_source='id_to_labels.csv'):
        super().__init__(data_source)
        self.sample_idx = sample_idx
        self.df_images = pd.read_csv(data_source)
        
    def __iter__(self):
        image_ids = self.df_images['image_id'].loc[self.sample_idx]
        #image_ids=self.sample_idx
        return iter(image_ids)
    
    def __len__(self):
        return len(self.sample_idx)
    


class ImageBatchSampler(BatchSampler):
    def __init__(self, 
                 sampler,
                 aug_count=5,
                 batch_size=30,
                 drop_last=True):
        super().__init__(sampler, batch_size, drop_last)
        self.aug_count = aug_count
        assert self.batch_size % self.aug_count == 0, 'Batch size must be an integer multiple of the aug_count.'
        
    def __iter__(self):
        batch = []
        
        for image_id in self.sampler:
            for i in range(self.aug_count):
                batch.append(image_id)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
############################################################################################################
def create_split_loaders(dataset_train,dataset_val, split, aug_count, batch_size):
    train_folds_idx = split[0]
    valid_folds_idx = split[1]
    train_sampler = ImageSampler(train_folds_idx)
    valid_sampler = ImageSampler(valid_folds_idx)
    train_batch_sampler = ImageBatchSampler(train_sampler, 
                                            aug_count, 
                                            batch_size)
    valid_batch_sampler = ImageBatchSampler(valid_sampler, 
                                            aug_count=1, 
                                            batch_size=batch_size,
                                            drop_last=False)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_sampler=train_batch_sampler)
    valid_loader = DataLoader(dataset_val, batch_sampler=valid_batch_sampler)
    return (train_loader, valid_loader)    


############################################################################################################3

def get_all_split_loaders(dataset_train,dataset_val, cv_splits, aug_count=5, batch_size=10):
    """Create DataLoaders for each split.

    Keyword arguments:
    dataset -- Dataset to sample from.
    cv_splits -- Array containing indices of samples to 
                 be used in each fold for each split.
    aug_count -- Number of variations for each sample in dataset.
    batch_size -- batch size.
    
    """
    split_samplers = []
    
    for i in range(len(cv_splits)):
        split_samplers.append(
            create_split_loaders(dataset_train,dataset_val,
                                 cv_splits[i], 
                                 aug_count, 
                                 batch_size)
        )
    return split_samplers
###########################################################################



def get_k_fold_loader(dir_name, binary_dir,label_csv_name,label_fine, batch_size, level, k=10, test=False):
    ids, labels = get_idx(dir_name, binary_dir)
    
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    
    splits = []
    for train_idx, test_idx in splitter.split(ids, labels):
        splits.append((train_idx, test_idx))
    
    shape = (300,300)
    
    mean = 0.42 # set  according to mean and std of the dataset
    std = 0.21
    
    if level=='up': #or level=='up_ant'or level=='up_post':
        mean = 0.27
        std = 0.21
        
    if level=='down':#or level=='down_ant'or level=='down_post':
        mean = 0.34#0.29
        std = 0.20
        
    # if level=='upa':
    #     mean = 0.27#0.18
    #     std = 0.21
        
    # if level=='upb':
    #     mean = 0.27#0.178
    #     std = 0.21
        
    # if level=='downa':
    #     mean = 0.181
    #     std = 0.209  
        
    # if level=='downb':
    #     mean = 0.25
    #     std = 0.207  
        
    
    dataset_train = DicomDataset(label_csv_name,label_fine, level,
                                 

                                 
                                          transforms=transforms.Compose([
                                                       
                                        transforms.ToPILImage(),
    
                                       
                                        transforms.RandomApply(torch.nn.ModuleList([
                                                        transforms.RandomAffine(degrees=10, translate=(
                                                            0.1, 0.1), scale=(.9, 1.1), shear=(0.01, 0.03)),
                                                        transforms.RandomAffine(
                                                            degrees=5),
                                                        transforms.RandomAffine(
                                                            degrees=0, translate=(0.1, 0.1)),
                                                        transforms.RandomAffine(
                                                            degrees=0, scale=(.9, 1.1)),
                                                        transforms.RandomAffine(
                                                            degrees=0, shear=(0.005, 0.02)),
                                                    ]), p=0.5),
                                        
                                        transforms.Resize(shape[0], interpolation=Image.NEAREST),
                                      
                                        transforms.ToTensor(),                           # convert the PIL Image to a tensor
                                        transforms.Normalize(mean,std)
    
                                                ]))
    
   
    if test==False:
        dataset_val = DicomDataset(label_csv_name,label_fine, level,
                                   transforms=transforms.Compose([transforms.ToPILImage(),
                                   transforms.Resize( shape[0], interpolation=Image.NEAREST),
                                   transforms.ToTensor(),                           # convert the PIL Image to a tensor
                                   transforms.Normalize(mean,std) ]))
    else:
        dataset_val = DicomDataset(label_csv_name,label_fine,level,
                                   transforms=transforms.Compose([transforms.ToPILImage(),
                                   transforms.Resize( shape[0], interpolation=Image.NEAREST),
                                   transforms.ToTensor(),                           # convert the PIL Image to a tensor
                                    ]))
     
    test_dataset = DicomDataset(label_csv_name,label_fine, level,
                                transforms=transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize( shape[0], interpolation=Image.NEAREST),
                                transforms.ToTensor(),                           # convert the PIL Image to a tensor
                                transforms.Normalize(mean,std) ]))    

    all_loader=[]

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    data = get_all_split_loaders(dataset_train, dataset_val, splits, aug_count=1, batch_size=batch_size)
    

    acc_list=[]
    sen_list=[]
    spec_list=[]
    acc_all_list=[]
    
        
                        
        
    return data        