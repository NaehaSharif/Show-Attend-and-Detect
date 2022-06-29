#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 14:22:03 2022

@author: naeha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 12:33:37 2022

@author: naeha
"""

from plotting_errors import plot_corr
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import csv 
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
import scipy.stats  
import pandas as pd
import csv 
import numpy as np
import pandas as pd
import re
import shutil
import os
import matplotlib.pyplot as plt
import pingouin as pg
from scipy import stats
import seaborn as sns
import csv
from get_metrics import *


file_ant='attempt/ant/fine_scores'
files_ant=os.listdir(file_ant)

file_post='attempt/post/fine_scores'
files_post=os.listdir(file_post)

folder_combined='attempt/combined/fine_scores'
folder_combined_sum='attempt/combined/scores'


def convert_row_to_list(r):
    out=r.replace("[", "")
    out=out.replace("]", "")
    out_list=[int(i) for i in out.split()]
    return out_list

pred_dic_all={}
pred_dic={}
pred_ant=[]
y_ant=[]
pred_post=[]
y_post=[]
id_ant=[]
id_post=[]
for name_ant in files_ant:
    pred_dic={}
    if name_ant.endswith(".csv"):
        print('*****************Results for', name_ant)
            
        with open(os.path.join(file_ant,name_ant))as csv_file:
            csv_reader = csv.reader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count>0:
                    
                    out=convert_row_to_list(row[1])                   
                    gt=convert_row_to_list(row[2])   
                    
                    y_ant.append(gt)
                    pred_ant.append(out)
                    id_ant.append(row[0])
                line_count+=1    
        name_post='post'+name_ant[3:] 
        #name_post='up_post'+name_ant[6:]    #in case of 4 splits
        with open(os.path.join(file_post,name_post))as csv_file:
            csv_reader = csv.reader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count>0:
                    
                    out=convert_row_to_list(row[1])                   
                    gt=convert_row_to_list(row[2])  
                    id_post.append(row[0])
                    
                    y_post.append(gt)
                    pred_post.append(out)
                line_count+=1   
if id_ant!=id_post:
    print('ids_dont_match')
y_ant=np.array(y_ant)  
pred_ant=np.array(pred_ant) 
y_post=np.array(y_post)  
pred_post=np.array(pred_post) 

combined_pred=pred_ant+pred_post
combined_gt=y_ant+ y_post

# saving fine _outs#############
outs = {'val_id': id_ant,'val_pred': list(combined_pred), 'val_gt': list(combined_gt)}
df = pd.DataFrame(outs, columns=['val_id','val_pred', 'val_gt'])
df.to_csv(os.path.join(folder_combined,"combined_outs.csv"))
##########################################################3


# saving total outs####################################
combined_predsum=np.sum(combined_pred, axis=1)
combined_gtsum=np.sum(combined_gt, axis=1)


outs = {'val_id': id_ant,'val_pred': list(combined_predsum), 'val_gt': list(combined_gtsum)}
df = pd.DataFrame(outs, columns=['val_id','val_pred', 'val_gt'])



df.to_csv(os.path.join(folder_combined_sum,"combined_outs.csv"))