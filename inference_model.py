#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 17:53:48 2022

@author: ecu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 14:22:39 2021

@author: naeha
"""

import numpy as np
import os
import glob
import torch
import torchvision
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms, utils
from skimage.transform import resize
import collections
import time
import tqdm
import copy
from pycocotools.coco import COCO
import math
import numpy as np
import torch.utils.data as data
from sklearn import  metrics
import pandas as pd
from torch.utils.data import Sampler
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import visdom
from sklearn import metrics
from sklearn.metrics import roc_auc_score
seed = 3
import torchvision.transforms as transforms
torch.manual_seed(seed)
from sklearn.metrics import confusion_matrix
import scipy.stats 
from efficientnet_pytorch import EfficientNet
import matplotlib.pyplot as plt
import json
from sklearn.metrics import cohen_kappa_score
from pytorchtools import EarlyStopping  
import scipy.stats      


from model import EncoderCNN, DecoderRNN
from utils import get_batch_caps, get_hypothesis
from plot import *
from plotting_errors import *
from get_loaders_split import * 
from train_val_func import train, validate
from eval_ import*
#from utils import visualize_attention

dir_name='/npy_folder-0' # path of the file where images are present
binary_dir='ids_to_score-bill.npy' # this is same for SE and DE because the ids are the same (the main irectory which links ids to scores)
# dont change this name#
label_csv_name='id_to_labels.csv' # this file contains the cummulative scores in csv format and is created by the k-fold split  (dont change this)
# dont change this name#
label_fine='ids_to_score-bill-fine.npy' # using this file means the sequence is only four numbers (divide the validation score accordingly)
## TODO #1: Select appropriate values for the Python variables below.
batch_size = 10         # batch size, change to 64
vocab_threshold = 3        # minimum word count threshold
vocab_from_file = False    # if True, load existing vocab file
embed_size = 256         # dimensionality of image and word embeddings
hidden_size = 512         # number of features in hidden state of the RNN decoder
num_features = 2048       # number of feature maps, produced by Encoder
num_epochs = 100            # number of training epochs
save_every = 1             # determines frequency of saving model weights
print_every = 5          # determines window for printing average loss
learning_rate=1e-4
# The size of the vocabulary.
vocab_size = 4
level='post'
device_ids=list(range(torch.cuda.device_count()))

model_folder='attempt/post/model/' # set the path accordingly
model_vis='model/attempt/post/vis/'

log_train = 'training_log.txt'       # name of files with saved training loss and perplexity
log_val = 'validation_log.txt'
bleu = 'bleu.txt'


# Build data loader.


flag=1
flag2=1

if flag==1:

    def visualize_attention(orig_image, outputs, atten_weights,name):
        """Plots attention in the image sample.
        
        Arguments:
        ----------
        - orig_image - image of original size
        - words - list of tokens
        - atten_weights - list of attention weights at each time step 
        """
        fig = plt.figure(figsize=(20,10)) 
        atten_weights= atten_weights.cpu().detach().numpy().copy()
        words=outputs.max(dim=-1).indices.cpu().detach().numpy().copy()
        len_tokens = len(words)
        img_ = np.load(name, allow_pickle=True).astype(np.float32)
        id_=name.split('/')[-1][:-4]
        L=['L1',  'L2', 'L3', 'L4']
        colr=['turbo']#, 'coolwarm', 'summer','gist_earth', 'rainbow']
        number=id_+'['
        for i in range(len(words)-1):
            atten_current = atten_weights[i]
            atten_current = atten_current.reshape(29,10)    #    reshape(7,7) 
            #ax = fig.add_subplot(len_tokens//2, len_tokens//2, i+1)
            ax = fig.add_subplot(1, len_tokens-1, i+1)
            #ax = fig.add_subplot(0, i)
            ax.set_title(L[i]+'= ' +str(words[i]))
            number=number+str(words[i])
            #img = ax.imshow(np.squeeze(orig_image.cpu().detach().numpy().copy()))
            img = ax.imshow(img_)
            ax.imshow(img_,cmap=plt.cm.gray)
            ax.imshow(atten_current, cmap=plt.get_cmap(colr[0]), alpha=0.3, extent=img.get_extent(), interpolation = 'gaussian')
            #print(outputs)
        number=number+']'
        plt.suptitle(number,fontsize=10)
        plt.tight_layout()
        
        #plt.show()
        save=model_vis+id_+'.png'
        plt.savefig(save)
        plt.close(fig)
    # Build data loader.
    data = get_k_fold_loader(dir_name, binary_dir,label_csv_name,label_fine,batch_size, level, k=10, test=False)
    print('Data Loaded')


    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    # Initialize the encoder and decoder. 
    encoder = EncoderCNN()
    decoder = DecoderRNN(num_features = num_features, 
                             embedding_dim = embed_size, 
                             hidden_dim = hidden_size, 
                             vocab_size = vocab_size,
                             device=device)
        
   
    decoder.eval()
    encoder.eval()
    
    # Move models to GPU if CUDA is available. 
    
    # when doing beam search use cpu
    #device = torch.device("cpu") #when only CPU
    
    
    input_word = torch.tensor(5).unsqueeze(0).to(device)


    mean = 0.42
    std = 0.21
    for curr_fold, (train_batch_loader, valid_batch_loader) in enumerate(data):
           
           
            train_data_loader=train_batch_loader
            val_data_loader=valid_batch_loader
    
            
            total_step_valid = len(val_data_loader) 
            total_step_train = len(train_data_loader) 
            
            
            encoder_file = model_folder+ level+'_encoder-'+str(curr_fold)+'_4.pkl' # set the path accordingly
            decoder_file = model_folder+ level+'_decoder-'+str(curr_fold)+'_4.pkl' 
            
            # Load the trained weights.
            encoder.load_state_dict(torch.load(encoder_file))
            decoder.load_state_dict(torch.load(decoder_file))

            
            # Move models to GPU if CUDA is available.
            
            encoder.to(device)
            decoder.to(device) 
    
            #############################################
            
            # if curr_fold==0:
            #      break
                
    
            if flag2==1:
                print('fold: ',str(curr_fold), ' started' )
                for i_step,(val_images, labels,val_captions, name) in enumerate(val_data_loader):
                    
                  # evaluation of encoder and decoder
                
                 
                 
                 
                 
                          val_captions_target = val_captions[:, 1:].to(device) 
                          val_captions = val_captions[:, :-1].to(device)
                          val_images = val_images.to(device)
                          
                         
                          features_val = encoder(val_images).to(device)
                          outputs_val, atten_weights_val = decoder(captions= val_captions,
                                                   features = features_val)
                          outputs= outputs_val.max(dim=-1).indices.cpu().detach().numpy().copy()
                          for n in range(len(val_images)):
                              atten_weights=atten_weights_val[n,:,:]
                              outputs=outputs_val[n]
                              orig_image=val_images[n][0]    
                              
                              
                              visualize_attention(orig_image,outputs, atten_weights, name[n])
                              gt=labels[n]
                              #print(gt, outputs.max(dim=-1).indices.cpu().detach().numpy().copy())
                              
                print('fold: ',str(curr_fold), ' done' )
                 
        ##############################################         

