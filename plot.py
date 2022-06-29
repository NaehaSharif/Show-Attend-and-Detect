#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 12:40:51 2021

@author: naeha
"""

import matplotlib.pyplot as plt
# line 1 points
def plotting_(no_of_epochs, train_acc, train_loss, val_acc, val_loss):
    
    fig, axs = plt.subplots(2)
    x1 = list(range(no_of_epochs))

    # plotting the line 1 points 
    axs[0].plot(x1, train_acc, label = "train_acc")
    # line 2 points
    axs[0].plot(x1, val_acc, label = "val_acc")
    # plotting the line 2 points 
    axs[0].legend()
    axs[0].set(xlabel='epochs - axis', ylabel='accuracy')
    # Set the y axis label of the current axis.

    # line 2 points
    # plotting the line 2 points 
    axs[1].plot(x1, train_loss, 'tab:red', label = "train_loss")
    axs[1].plot(x1, val_loss, 'tab:green',label = "val_loss")
    axs[1].set(xlabel='epochs - axis', ylabel='loss')

    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()
