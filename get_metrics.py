#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:47:34 2021

@author: naeha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 13:21:07 2021

@author: naeha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 10:02:05 2021
reads the json files containing scores and plots the graphs
@author: naeha
"""

from plotting_errors import plot_corr
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import csv 
import tqdm
import copy
import glob, os, sys, warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
from sklearn.metrics import cohen_kappa_score    
    
from plotting_errors import plot_corr
import os
import json
import numpy as np
import matplotlib.pyplot as plt

import argparse
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score
#from torchsampler import ImbalancedDatasetSampler
seed = 3

from sklearn.metrics import confusion_matrix
import scipy.stats         
from plotting_errors import *
import matplotlib.pyplot as plt
#import pingouin as pg
from scipy import stats
import seaborn as sns



gt_dic={}

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.tight_layout()   


        


    
def plot_acc_sen_spec(x,y,z):
    acc= np.array(x)
    sen= np.array(y)
    spec= np.array(z)

    
    
    # Calculate the average
    acc_mean = np.mean(acc)
    sen_mean = np.mean(sen)
    spec_mean = np.mean(spec)
    
    # Calculate the standard deviation
    acc_std = np.std(acc)
    sen_std = np.std(sen)
    spec_std = np.std(spec)
    
    # Define labels, positions, bar heights and error bar heights
    labels = ['accuracy', 'sensitivity', 'specificity']
    x_pos = np.arange(len(labels))
    CTEs = [acc_mean, sen_mean, spec_mean]
    error = [acc_std, sen_std, spec_std]

    print('Accuracy, {0:1f}, {1:1f}'.format(CTEs[0],error[0] ))
    print('Sensitivity, {0:1f},  {1:1f}'.format(CTEs[1],error[1] ))
    print('Specificity, {0:1f},  {1:1f}'.format(CTEs[2],error[2] ))
    return CTEs, error


def plot_npv_ppv(x,y):
    npv= np.array(x)
    ppv= np.array(y)


    
    
    # Calculate the average
    npv_mean = np.nanmean(npv)
    ppv_mean = np.nanmean(ppv)

    
    # Calculate the standard deviation
    npv_std = np.nanstd(npv)
    ppv_std = np.nanstd(ppv)


    CTEs = [npv_mean, ppv_mean]
    error = [npv_std, ppv_std]

    print('NPV, {0:1f}, {1:1f}'.format(CTEs[0],error[0] ))
    print('PPV, {0:1f}, {1:1f}'.format(CTEs[1],error[1] ))

    return CTEs, error


def plot_pll_nll_f1(x,y,z):
    pll= np.array(x)
    nll= np.array(y)
    f1score= np.array(z)


    
    
    # Calculate the average
    pll_mean = np.nanmean(pll)
    nll_mean = np.nanmean(nll)
    f_mean = np.nanmean(f1score)
    
    # Calculate the standard deviation
    pll_std = np.nanstd(pll)
    nll_std = np.nanstd(nll)
    f_std = np.nanstd(f1score)

    CTEs = [pll_mean, nll_mean, f_mean]
    error = [pll_std, nll_std, f_std]

    print('pll, {0:1f}, {1:1f}'.format(CTEs[0],error[0] ))
    print('nll, {0:1f}, {1:1f}'.format(CTEs[1],error[1] ))
    print('f1score, {0:1f},  {1:1f}'.format(CTEs[2],error[2] ))
    
    
    
    return CTEs, error

def get_everything(y, pred,name):
    y=np.array(y)  
    pred=np.array(pred)
     
    labels_class=[]
    for i in y:
        if i >= 6.0:
            labels_class.append(2)
        elif i < 2.0:
            labels_class.append(0) 
        else:
            labels_class.append(1)
    
    preds=[]
    for i in pred:
        if i >= 6.0:
            preds.append(2)
        elif i < 2.0:
            preds.append(0) 
        else:
            preds.append(1)        
    
    
    
    cnf_matrix = confusion_matrix(preds, labels_class,labels=[0,1, 2])
    kappa= confusion_matrix(preds, labels_class,labels=[0,1, 2])
    np.set_printoptions(precision=2)
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=['med', 'low', 'high'],
    #                       title='Confusion matrix, without normalization')
    
    #############################################################################
    # https://towardsdatascience.com/multi-class-classification-extracting-performance-metrics-from-the-confusion-matrix-b379b427a872
    FP = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    
    
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    
    # # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)
    
    recall=(TP/(TP+FN+0.00001)) *100
    specificity=(TN/(TN+FP+0.00001)) *100
    #corr, _ = pearsonr(l_c, o_c)
    corr, p = scipy.stats.pearsonr(y, pred)
      
    rho, sp = scipy.stats.spearmanr(y,pred)   
        
    
    plot_confusion_matrix(cnf_matrix, classes=['low', 'mod', 'high'],
                                   title='Confusion matrix')
    
    print('Acc: {:.3f} , pearson {:3f}, spearman {:3f},'.format(np.mean(ACC), corr, rho))
    print('*** Class Accuracy***')
    print('(Low, moderate, high):  {:2f}, {:2f}, {:2f}'.format(ACC[0], ACC[1], ACC[2] ))
    print('***Class# Low****')
    print('(tp, fp, tn, fn, sensitivity, specificity):  {:2f}, {:2f}, {:2f}, {:2f}, {:2f}, {:2f}'.format( TP[0], FP[0], TN[0], FN[0], recall[0], specificity[0] ))
    print('***Class# Medium***')
    print('(tp, fp, tn, fn, sensitivity, specificity):  {:2f}, {:2f}, {:2f}, {:2f}, {:2f}, {:2f}'.format( TP[1], FP[1], TN[1], FN[1], recall[1], specificity[1] ))
    print('***Class# High****')
    print('(tp, fp, tn, fn, sensitivity, specificity):  {:2f}, {:2f}, {:2f}, {:2f}, {:2f}, {:2f}'.format( TP[2], FP[2], TN[2], FN[2], recall[2], specificity[2] ))
    
    
       
    print('*** Class NPV***')
    print('(Low, moderate, high):  {:2f}, {:2f}, {:2f}'.format(NPV[0], NPV[1], NPV[2] ))
    print('*** Class PPV***')
    print('(Low, moderate, high):  {:2f}, {:2f}, {:2f}'.format(PPV[0], PPV[1], PPV[2] ))
    
    
    
    #######################################################
    
def get_back_everything(y, pred):
    y=np.array(y)  
    pred=np.array(pred)
     
    labels_class=[]
    for i in y:
        if i >= 6.0:
            labels_class.append(2)
        elif i < 2.0:
            labels_class.append(0) 
        else:
            labels_class.append(1)
    
    preds=[]
    for i in pred:
        if i >= 6.0:
            preds.append(2)
        elif i < 2.0:
            preds.append(0) 
        else:
            preds.append(1)        
    
    
    
    cnf_matrix = confusion_matrix(preds, labels_class,labels=[0,1, 2])
    kappa= confusion_matrix(preds, labels_class,labels=[0,1, 2])
    np.set_printoptions(precision=2)
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=['med', 'low', 'high'],
    #                       title='Confusion matrix, without normalization')
    
    #############################################################################
    # https://towardsdatascience.com/multi-class-classification-extracting-performance-metrics-from-the-confusion-matrix-b379b427a872
    FP = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    
    
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    

    # # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)
    
    recall=(TP/(TP+FN+0.00001)) *100
    specificity=(TN/(TN+FP+0.00001)) *100
    #corr, _ = pearsonr(l_c, o_c)
    corr, p = scipy.stats.pearsonr(y, pred)
      
    rho, sp = scipy.stats.spearmanr(y,pred)   
        
    
    return ACC, recall, specificity, NPV,PPV, corr, rho,TP
    