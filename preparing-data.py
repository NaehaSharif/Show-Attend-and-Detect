#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 13:09:26 2021

@author: naeha


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""This script reads the csv file containing the ids, score and dates of the dicoms  and  then creates a numpy
containing the names to score mapping. """


import csv 
import numpy as np
import pandas as pd
import re
import shutil
import os
import matplotlib.pyplot as plt




csv_name='/SR_LW_VFA_CaseNames_1100_final_Batch1.xls'

#en file in read mode
df = pd.read_excel(csv_name)
df.dropna(axis="columns", how="any")
IDS = pd.DataFrame(df, columns= ['CaseName'])
Score= pd.DataFrame(df, columns= ['SR_SUM'])

IDS=list(IDS['CaseName'])
score=list(Score['SR_SUM'])
Score= pd.DataFrame(df, columns= ['L1 Ant', 'L1 Post', 'L2 Ant', 'L2 Post', 'L3 Ant','L3 Post', 'L4 Ant', 'L4 Post' ])

all_scores=np.array(list(zip([5]*len(IDS),list(Score['L1 Ant']), list(Score['L1 Post']),list(Score['L2 Ant']), list(Score['L2 Post']) ,
                    list(Score['L3 Ant']), list(Score['L3 Post']), list(Score['L4 Ant']), list(Score['L4 Post']), [6]*len(IDS))))

updated_array = np.nan_to_num(all_scores)


Id_to_score=dict(zip(IDS,updated_array))


    
         
cars = {'image_id': list(Id_to_score.keys()),
        'labels': list(Id_to_score.values())
        }

df = pd.DataFrame(cars, columns= ['image_id', 'labels'])


df.to_csv ('id_to_score-bill-fine.csv', index = False, header=True) 


np.save("ids_to_score-bill-fine.npy",  Id_to_score) 
# 
