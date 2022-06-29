#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 15:18:06 2021

@author: naeha
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_error(x,y):
    val= np.array(x)
    test= np.array(y)
    
    
    
    # Calculate the average
    val_mean = np.mean(val)
    test_mean = np.mean(test)
    
    
    # Calculate the standard deviation
    val_std = np.std(val)
    test_std = np.std(test)
    
    
    # Define labels, positions, bar heights and error bar heights
    labels = ['validation', 'test']
    x_pos = np.arange(len(labels))
    CTEs = [val_mean, test_mean]
    error = [val_std, test_std]
    
    
    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs,
           yerr=error,
           align='center',
           alpha=0.5,
           ecolor='black',
           capsize=10)
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title('Performance on validation and test set')
    ax.yaxis.grid(True)
    
    
    # Save the figure and show
    plt.tight_layout()
    plt.savefig('bar_plot_with_error_bars.png')
    plt.show()





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
    
    
    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs,
           yerr=error,
           align='center',
           alpha=0.5,
           ecolor='black',
           capsize=10)
    ax.set_ylabel('Performance (%age)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title('10-fold Cross Validation')
    ax.yaxis.grid(True)
    
    
    # Save the figure and show
    plt.tight_layout()
    plt.savefig('cross validation.png')
    plt.show()
    
    print('Accuracy: mean {0:1f}, std {1:1f}'.format(CTEs[0],error[0] ))
    print('Sensitivity: mean {0:1f}, std {1:1f}'.format(CTEs[1],error[1] ))
    print('Specificity: mean {0:1f}, std {1:1f}'.format(CTEs[2],error[2] ))
    
    
    
def plot_corr(x,y):
    acc= np.array(x)
    sen= np.array(y)
   

    
    
    # Calculate the average
    acc_mean = np.mean(acc)
    sen_mean = np.mean(sen)
    
    # Calculate the standard deviation
    acc_std = np.std(acc)
    sen_std = np.std(sen)
   
    
    # Define labels, positions, bar heights and error bar heights
    labels = ['pearson', 'spearman']
    x_pos = np.arange(len(labels))
    CTEs = [acc_mean, sen_mean]
    error = [acc_std, sen_std]
    
    
    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs,
           yerr=error,
           align='center',
           alpha=0.5,
           ecolor='black',
           capsize=10)
    ax.set_ylabel('Correlation')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_title('10-fold Cross Validation')
    ax.yaxis.grid(True)
    
    
    # Save the figure and show
    plt.tight_layout()
    plt.savefig('cross validation.png')
    plt.show()
    
    print('Pearson: mean {0:1f}, std {1:1f}'.format(CTEs[0],error[0] ))
    print('Spearman: mean {0:1f}, std {1:1f}'.format(CTEs[1],error[1] ))
  