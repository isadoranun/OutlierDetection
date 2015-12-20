
import numpy as np
import scipy as sp
import math
#import matplotlib.pyplot as plt
#import matplotlib.pylab as pl
#import seaborn as sns
import csv
import copy
import glob
import pickle
import os.path

#for f in /n/regal/TSC/Catalina/*;do sbatch /n/regal/TSC/Extract_features/run_extractF_Catalina.sh  "$readlink -f "$f"" ; done;


def concatenate_results(fileregex):

    count = 0
    for filename in os.listdir(fileregex):
        if filename.endswith('periods.pk'):
            base = fileregex+filename.split('_')[0]+'_'+filename.split('_')[1]+'_'+filename.split('_')[2]
             
            if (count == 0 and  os.stat(fileregex+filename).st_size != 0 and os.path.isfile(base+'_ids.pk') and os.stat(base+'_ids.pk').st_size != 0  and os.path.isfile(base+'_Scores.pk') and os.stat(base+'_Scores.pk').st_size != 0 and os.path.isfile(base+'_X_out_I.pk') and os.stat(base+'_X_out_I.pk').st_size != 0):
                print 'holi'    
                F_periods = pickle.load(open(fileregex + filename, 'rb'))
                Scores = pickle.load(open(base+'_Scores.pk', 'rb'))
                X_out_I = pickle.load(open(base+'_X_out_I.pk', 'rb'))
                F_ids =  pickle.load(open(base+'_ids.pk', 'rb'))

                count = count + 1;


            elif os.stat(fileregex+filename).st_size != 0 and os.path.isfile(base+'_ids.pk') and os.stat(base+'_ids.pk').st_size != 0  and os.path.isfile(base+'_Scores.pk') and os.stat(base+'_Scores.pk').st_size != 0 and os.path.isfile(base+'_X_out_I.pk') and os.stat(base+'_X_out_I.pk').st_size != 0:
                try:
                	F_periods = np.concatenate((F_periods, pickle.load(open(fileregex + filename, 'rb'))))
                except:
                	pass
                try:
                	F_ids = np.concatenate((F_ids, pickle.load(open(base+'_ids.pk', 'rb'))))
                except:
                	F_periods = np.delete(F_periods,(len(F_periods)-1), axis=0)
                	pass
               	try:
                	Scores = np.concatenate((Scores, pickle.load(open(base+'_Scores.pk', 'rb'))))
                except:
                	F_periods = np.delete(F_periods,(len(F_periods)-1),axis=0)
                	F_ids = np.delete(F_ids,(len(F_ids)-1),axis=0)
                	pass
                try:
                	X_out_I = np.concatenate((X_out_I, pickle.load(open(base+'_X_out_I.pk', 'rb'))))
                except:
                	F_periods = np.delete(F_periods,(len(F_periods)-1),axis=0)
                	F_ids = np.delete(F_ids,(len(F_ids)-1),axis=0)
                	Scores = np.delete(Scores,(len(Scores)-1),axis=0)
                	pass
                count = count + 1 
    return F_ids, F_periods, Scores, X_out_I

            

# def concatenate_periods_from_cluster(fileregex):
#     F_periods = []
#     for filename in os.listdir(fileregex):
#         if filename.endswith('periods.pk'):
#             F_periods = np.concatenate((F_periods, pickle.load(open(fileregex + filename, 'rb'))))
#     return F_periods

# def concatenate_ids_from_cluster(fileregex):
#     F_ids = []
#     for filename in os.listdir(fileregex):
#         if filename.endswith('ids.pk'):
#              F_ids = np.concatenate((F_ids, pickle.load(open(fileregex + filename, 'rb'))))
#     return F_ids

# def concatenate_scores_from_cluster(fileregex):
#     count = 0
#     for filename in os.listdir(fileregex):
#         if filename.endswith('Scores.pk'):
#             count = count + 1;
#             if count == 1:
#                 Scores = pickle.load(open(fileregex + filename, 'rb'))
#             else:    
#                 Scores = np.concatenate((Scores, (pickle.load(open(fileregex + filename, 'rb')))),axis=0)
#     return Scores

# def concatenate_X_from_cluster(fileregex):
#     count = 0
#     for filename in os.listdir(fileregex):
#         if filename.endswith('X_out_I.pk'):
#             count = count + 1;
#             if count == 1 :
#                 X_out_I = pickle.load(open(fileregex + filename, 'rb'))
#             else:
#                 X_out_I = np.concatenate((X_out_I, pickle.load(open(fileregex + filename, 'rb'))))
#     return X_out_I

