#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:25:02 2021

* computes the mean and std of the images *

@author: naeha
"""
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd


R_total = 0
G_total = 0
B_total = 0

total_pixel = 0




filepath='npy_folder/'
allfiles=[]
s=0
for i in [0]:
    new=filepath +'-'+str(i)
    
    pathDir=os.listdir(new)
    allfiles+=pathDir

    
    for idx in range(len(pathDir)):
        
        filename = pathDir[idx]
        # if filename[:-4]+'.png' in ids:
        img = np.load(os.path.join(new, filename),mmap_mode=None, allow_pickle=True, fix_imports=True)
        img= np.array(img, dtype=np.float32)
      
    
        R_total = R_total + np.sum(img[:,:] )
        G_total = G_total + np.sum(img[:,:] )
        B_total = B_total + np.sum(img[:,:])
        



total_count=len(allfiles)*np.shape(img)[0]*np.shape(img)[1]

# total_count=s*np.shape(img)[0]*np.shape(img)[1]
R_mean=R_total/total_count
G_mean=G_total/total_count
B_mean=B_total/total_count

R_total = 0
G_total = 0
B_total = 0

for i in [0]:
    new=filepath +'-'+str(i)
    
    pathDir=os.listdir(new)
    
    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        
        # if filename[:-4]+'.png' in ids:
        img = np.load(os.path.join(new, filename))
        img= np.array(img, dtype=np.float32)
        total_pixel = total_pixel + img.shape[0] * img.shape[1]
    
        R_total = R_total + np.sum((img[:,:]  - R_mean) ** 2)
        G_total = G_total + np.sum((img[:,:]  - G_mean) ** 2)
        B_total = B_total + np.sum((img[:,:] - B_mean) ** 2)

R_std = np.sqrt(R_total / total_count)
G_std = np.sqrt(G_total / total_count)
B_std = np.sqrt(B_total / total_count)

