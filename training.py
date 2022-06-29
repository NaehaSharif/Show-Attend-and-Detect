#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 08:41:08 2022

@author: ecu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 17:36:35 2022

@author: ecu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 12:07:14 2021

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

########## user defined###############
from model import EncoderCNN, DecoderRNN
from utils import get_batch_caps, get_hypothesis
from get_loaders_split import * 
from train_val_func import train, validate
from eval_ import*
from plot import *
from plotting_errors import *




start_time = time.time()

dir_name='npy_folder-0/' # path of the file where images are present
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
num_features = 2048       # number of feature maps, produced by Encoder (when in case of effnet b3)
num_epochs = 100            # number of training epochs
save_every = 1             # determines frequency of saving model weights
print_every = 5          # determines window for printing average loss
learning_rate=1e-4
# The size of the vocabulary.
vocab_size = 8
level='ant'
device_ids=list(range(torch.cuda.device_count()))

log_train = 'training_log.txt'       # name of files with saved training loss and perplexity
log_val = 'validation_log.txt'
bleu = 'bleu.txt'


# Build data loader.
data = get_k_fold_loader(dir_name, binary_dir,label_csv_name,label_fine,batch_size, level, k=10, test=False)
print('Data Loaded')





# Open the training log file.
file_train = open(log_train, 'w')
file_val = open(log_val, 'w')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


for curr_fold, (train_batch_loader, valid_batch_loader) in enumerate(data):
        

        # Initialize the encoder and decoder. 
        encoder = EncoderCNN()
        decoder = DecoderRNN(num_features = num_features, 
                             embedding_dim = embed_size, 
                             hidden_dim = hidden_size, 
                             vocab_size = vocab_size,
                             device=device)
        
    
        # Move models to GPU if CUDA is available. 
        
        # device = torch.device("cpu") #when only CPU
        
        encoder.to(device)
        decoder.to(device)
        
        # Define the loss function with weights.
        
        if level=='ant': 
             weights = torch.tensor([1., 4.4, 10.7, 10.4]).to(device)
            
        elif level=='post': 
             weights = torch.tensor([1., 4.1, 8.1, 7.4]).to(device)
        

            
            
            
        criterion = nn.CrossEntropyLoss(weight=weights,reduction='mean').to(device)
                
        
       
        params = list(decoder.parameters())
        #params_en= list(encoder.parameters())
        
        # TODO #4: Define the optimizer.
        optimizer = torch.optim.Adam(params, lr = learning_rate)
       
        train_data_loader=train_batch_loader
        val_data_loader=valid_batch_loader

        
        total_step_valid = len(val_data_loader) 
        total_step_train = len(train_data_loader) 
        ########################################################################




        # store BLEU scores in list 
        bleu_scores = []
        patience=30
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        history = {'val_loss':[], 'val_acc':[],'val_perplex':[], 'train_perplex':[], 'val_pear':[], 'val_kend':[]}
        best_acc = 0.0
        best_loss=500.0
        best_pear=0.0
        model_name='seq2seq'
        
        for epoch in range(0, num_epochs+1):   
            
           
            train_loss=train(epoch, encoder, decoder, optimizer, criterion, total_step_train, num_epochs =num_epochs,
                  train_data_loader = train_data_loader,
                  write_file = file_train, 
                  save_every = 100, 
                  vocab_size=vocab_size,
                  device=device,
                  print_every=print_every)
            print('train one epoch')
            history["train_perplex"].append(train_loss)
            
            gt, pred,loss, fine_outs, fine_gt, ids,epoch_perp_avg=validate(epoch, encoder, decoder, optimizer, criterion, 
                     total_step = total_step_valid, 
                     num_epochs = num_epochs, 
                     val_data_loader = val_data_loader,
                     write_file=file_val,
                      vocab_size=vocab_size,
                      device=device,
                      print_every=print_every)
            epochtp, epochfp, epochtn, epochfn, recall, specificity, corr, p, rho, sp, epoch_acc_all, epoch_npv, epoch_ppv=evaluate(pred, gt)
            history["val_acc"].append((epoch_acc_all[1]+ epoch_acc_all[0]+ epoch_acc_all[2])/3.0)
            history["val_loss"].append(loss)
            history["val_perplex"].append(epoch_perp_avg)
            
            # evaluate fine pearson 
            f_outs=np.array(fine_outs)
            f_gt=np.array(fine_gt)
            _,n=f_outs.shape
            pear=0
            kend=0
     
            for i in range(0,n):
                pred_score=f_outs[:,i]
                gt_score=f_gt[:,i] 
                common=(gt_score==pred_score)
                Pear, p=scipy.stats.pearsonr(pred_score, gt_score)
                tau, p_value = scipy.stats.kendalltau(pred_score, gt_score)
                #print(Pear)
                pear+=Pear
                kend+=tau
                
            history['val_pear'].append(pear)
            history['val_kend'].append(kend)
                        
            if  best_pear < pear:
                        best_pear = pear
                        best_acc = (epoch_acc_all[1]+ epoch_acc_all[0]+ epoch_acc_all[2])/3.0
                        btp, bfp, btn, bfn, bsen, bspec, bcorr, bp,brho, bsp, bacc_all, b_npv, b_ppv=epochtp, epochfp, epochtn, epochfn, recall, specificity, corr, p, rho, sp, epoch_acc_all, epoch_npv, epoch_ppv
                        best_gt, best_pred=gt, pred
                        best_fine_outs, best_fine_gt=fine_outs, fine_gt
                        
                        torch.save(decoder.state_dict(),  '%s_decoder-%d_4.pkl' % (level,curr_fold))
                        torch.save(encoder.state_dict(), '%s_encoder-%d_4.pkl' % (level,curr_fold))
                        stop=epoch
                        
            
            early_stopping(-history["val_pear"][-1],decoder)
                
            if early_stopping.early_stop or (epoch==num_epochs):
                    print("Early stopping")
                    
                
                    
                    outs = {'val_ids': list(ids),'val_pred': list(best_pred), 'val_gt': list(best_gt)}
                    outs2=  {'val_ids': list(ids),'val_pred': list(best_fine_outs), 'val_gt': list(best_fine_gt)}
                 
                    df = pd.DataFrame(outs, columns=['val_ids','val_pred', 'val_gt'])
                    df_fine = pd.DataFrame(outs2, columns=['val_ids','val_pred', 'val_gt'])
                    
                    df.to_csv("%s_%.4facc_%s_sen-%.2f_spec-%.3ffine_corr-%.3f_epoch-%.2f_pv-%.2f_rho-%.2f_prho.csv"%(str(curr_fold), stop, level, bsen[2], bsen[2],best_pear,epoch, bp,brho),
                                   index=False, header=True)
                    df_fine.to_csv("%s_fine_outs%s_4.csv"%(level,str(curr_fold), ),
                                   index=False, header=True)
                    
                    val_pear=history["val_pear"][:epoch]
                    val_kend=history["val_kend"][:epoch]
                    train_loss=history["train_perplex"][:epoch]
                    val_plex=history["val_perplex"][:epoch]
                    
                    x1 = list(range(epoch))
                    plt.plot(x1, val_plex, 'tab:green',label = "val_perplex")
                    plt.plot(x1, val_pear, 'tab:blue',label = "val_pearson")
                    plt.plot(x1, val_kend, 'tab:orange',label = "val_kendall")
                    plt.plot(x1, train_loss, 'tab:red', label = "train_perplex")
                    # show a legend on the plot
                    plt.legend()
                    # Display a figure.
                    plt.show()
                    
                    
                    break 
        
            
print("--- %s seconds ---" % (time.time() - start_time))