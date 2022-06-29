#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 12:36:05 2021

@author: naeha
"""
import scipy
import numpy as np
import os
import time
import tqdm
from sklearn import  metrics
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import scipy.stats         
import matplotlib.pyplot as plt
import json
from sklearn.metrics import cohen_kappa_score

def convert_to_labels(preds):
    preds1= (preds)>= 6
    preds2= (preds)<2
    
    preds1=preds1*2
    preds2=preds2*1
    preds3=preds*0
    
    
    preds = preds1+preds2+preds3
    
    return preds

def evaluate(pred, gt):    
    preds_a=np.array(pred).copy()
    labels_class_a=np.array(gt).copy()
    
    preds=convert_to_labels(preds_a)                        
    labels_class=convert_to_labels(labels_class_a)                           
    
    
    
    
    cnf_matrix = confusion_matrix(preds, labels_class,labels=[0,1, 2])
    kappa= cohen_kappa_score(preds, labels_class,labels=[0,1, 2])
    np.set_printoptions(precision=2)
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=['med', 'low', 'high'],
    #                       title='Confusion matrix, without normalization')
    
    #############################################################################
    # https://towardsdatascience.com/multi-class-classification-extracting-performance-metrics-from-the-confusion-matrix-b379b427a872
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
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
    
    
    ACC = (TP+TN)/(TP+FP+FN+TN)
    #######################################################################################################
    
    
    # statistics
    
    epochtp=TP
    epochfp=FP
    epochtn=TN
    epochfn=FN
    epoch_acc=ACC
    
    running_corrects= sum(preds == labels_class)
    epoch_ppv=PPV
    epoch_npv=NPV
    
        
    recall=(epochtp/(epochtp+epochfn+0.00001)) *100
    specificity=(epochtn/(epochtn+epochfp+0.00001)) *100
      
    corr, p = scipy.stats.pearsonr(pred, gt)
    
    rho, sp = scipy.stats.spearmanr(pred, gt)
    print('Val Acc: {:.3f} , pearson {:3f}'.format((running_corrects/ len(gt) *100), corr))
    print('*** Class Accuracy***')
    print('(Low, moderate, high):  {:2f}, {:2f}, {:2f}'.format(epoch_acc[1], epoch_acc[0], epoch_acc[2] ))
    print('***Class# Low****')
    print('(tp, fp, tn, fn, sensitivity, specificity):  {:2f}, {:2f}, {:2f}, {:2f}, {:2f}, {:2f}'.format( epochtp[1], epochfp[1], epochtn[1], epochfn[1], recall[1], specificity[1] ))
    print('***Class# Medium***')
    print('(tp, fp, tn, fn, sensitivity, specificity):  {:2f}, {:2f}, {:2f}, {:2f}, {:2f}, {:2f}'.format( epochtp[0], epochfp[0], epochtn[0], epochfn[0], recall[0], specificity[0] ))
    print('***Class# High****')
    print('(tp, fp, tn, fn, sensitivity, specificity):  {:2f}, {:2f}, {:2f}, {:2f}, {:2f}, {:2f}'.format( epochtp[2], epochfp[2], epochtn[2], epochfn[2], recall[2], specificity[2] ))
    
    return epochtp, epochfp, epochtn, epochfn, recall, specificity, corr, p, rho, sp, epoch_acc, epoch_npv, epoch_ppv

            # deep copy the model
            
