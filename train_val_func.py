#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 08:37:34 2022

@author: ecu
"""




import numpy as np
import sys
import os
import torch
import scipy.stats  

def train(epoch, 
          encoder, 
          decoder, 
          optimizer, 
          criterion,
          total_step,
          num_epochs, 
          train_data_loader, 
          write_file, 
          save_every,
          vocab_size,
          device,
          print_every):
    """ Train function for a single epoch. 
    Arguments: 
    ----------
    - epoch - number of current epoch
    - encoder - model's Encoder
    - decoder - model's Decoder
    - optimizer - model's optimizer (Adam in our case)
    - criterion - loss function to optimize
    - num_epochs - total number of epochs
    - data_loader - specified data loader (for training, validation or test)
    - write_file - file to write the training logs
    
    """
    epoch_loss = 0.0
    epoch_perplex = 0.0
    
        
    for i_step,(images, labels,captions, name) in enumerate(train_data_loader):
        
        # training mode on
        encoder.eval() # no fine-tuning for Encoder
        decoder.train()
       
        
        # target captions, excluding the first word
        captions_target = captions[:, 1:].to(device) 
        # captions for training without the last word
        captions_train = captions[:, :-1].to(device)
        
        # Move batch of images and captions to GPU if CUDA is available.
        images = images.to(device)
        
        # Zero the gradients.
        decoder.zero_grad()
        encoder.zero_grad()
        
        # Pass the inputs through the CNN-RNN model.
        features = encoder(images)
        outputs, atten_weights = decoder(captions= captions_train,
                                         features = features)
       
        # Calculate the batch loss.
        loss = criterion(outputs.view(-1, vocab_size), captions_target.reshape(-1))
        
      
        
        # Backward pass.
        loss.backward()
        
        # Update the parameters in the optimizer.
        optimizer.step()
        
        perplex = np.exp(loss.item())
        epoch_loss += loss.item()
        epoch_perplex += perplex
        
    
        
    epoch_loss_avg = epoch_loss / total_step
    epoch_perp_avg = epoch_perplex / total_step
    
    print('\r')
    print('Epoch train:', epoch)
    print('\r' + 'Avg. Loss train: %.4f, Avg. Perplexity train: %5.4f' % (epoch_loss_avg, epoch_perp_avg), end="")
    print('\r')
    
    # Save the weights.
    return epoch_perp_avg


def validate(epoch, 
             encoder, 
             decoder, 
             optimizer, 
             criterion, 
             total_step, num_epochs,
             val_data_loader,
             write_file, 
             vocab_size,
              device,
              print_every):
    """ Validation function for a single epoch. 
    Arguments: 
    ----------
    - epoch - number of current epoch
    - encoder - model's Encoder (evaluation)
    - decoder - model's Decoder (evaluation)
    - optimizer - model's optimizer (Adam in our case)
    - criterion - optimized loss function
    - num_epochs - total number of epochs
    - data_loader - specified data loader (for training, validation or test)
    - write_file - file to write the validation logs
    """
    epoch_loss = 0.0
    epoch_perplex = 0.0
    gt=[]
    all_outs=[]
    fine_outs=[]
    fine_gt=[]
    ids=[]
      
    for i_step,(val_images, labels,val_captions, name) in enumerate(val_data_loader):

        # evaluation of encoder and decoder
        encoder.eval()
        decoder.eval()
        
        
        val_captions_target = val_captions[:, 1:].to(device) 
        val_captions = val_captions[:, :-1].to(device)
        val_images = val_images.to(device)
        
        
        features_val = encoder(val_images)
        outputs_val, atten_weights_val = decoder(captions= val_captions,
                                         features = features_val)
        
        
        loss_val = criterion(outputs_val.view(-1, vocab_size), 
                              val_captions_target.reshape(-1))
        #loss_val =criterion(b,a)
        
        outputs_val=outputs_val.max(dim=-1).indices.cpu().detach().numpy().copy()
        all_outs.extend(list(np.sum(outputs_val[:, :-1], axis=1)))
        
        fine_outs.extend(outputs_val[:, :-1])
        fine_gt.extend(val_captions[:, 1:].cpu().detach().numpy().copy())
        ids.extend(name)
        
        gt.extend(list((labels.cpu().detach()).long().numpy()))
        
        
        
        perplex = np.exp(loss_val.item())
        epoch_loss += loss_val.item()
        epoch_perplex += perplex
  
    
    
    epoch_loss_avg = epoch_loss / total_step
    epoch_perp_avg = epoch_perplex / total_step
    print('\r')
    print('Epoch val:', epoch)
    print('\r' + 'Avg. Loss train: %.4f, Avg. Perplexity val: %5.4f' % (epoch_loss_avg, epoch_perp_avg), end="")
    print('\r')
            

    
    return gt, all_outs,epoch_loss_avg, fine_outs, fine_gt, ids,epoch_perp_avg

                                                                                       