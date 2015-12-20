#!/usr/bin/env python

import numpy as np
import scipy as sp
import pandas as pd
import math
import csv
import copy
import glob
import pickle
import os.path
import time
import sys

import knn
import rf_jp 
import loci
import LPD


#import ML routines
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
from sklearn import svm
from scipy import optimize
from sklearn.neighbors import KDTree
import statsmodels.api as sm

#import PCA stuff
from sklearn.decomposition import PCA

#Get training set 

def main(argv):

    #Get data to analize

    filename = argv;
    chip = filename.split('/')[len(filename.split('/'))-1].split('.csv')[0];



    data = pd.read_csv(filename ,index_col=0) 
    F = data.values
    F_periods = data.PeriodLS.values
    F_ids = data.index.values    


   # [F, F_ids, F_periods] = np.nan_to_num(np.array(lc)).astype('float'), np.array(y_id), (np.array(period)).astype('float')


    X_out_I = F;

    experts = [knn.knn, loci.loci,  LPD.LPD, rf_jp.rf_jp]
    #experts = [KNN1_wrapper, KNN2_wrapper, svm_jp, LOCI_wrapper, Eskin_wrapper]
    

    if knn.knn in experts:
        n_exp = len(experts)+1;
    else:
        n_exp = len(experts);
        
    Scores = np.zeros((len(X_out_I), n_exp))
    e = 0;

    for expert in experts:
        print e
        if expert == knn.knn:
            print expert
            t0=time.time()
            Scores[:,e], Scores[:,e+1] = expert(X_out_I)
            print time.time()-t0
            e = e + 2;
        else:
            print expert
            t0=time.time()
            #ayuda = rf_jp.rf_jp(X_out_I)
            #Scores[:,e] = ayuda
            #print "DOOOONEEEE"
            s = expert(X_out_I)
            Scores[:,e] = s
            #print "NOOOOO"
            #Scores[:,e] = expert(X_out_I)
            print time.time()-t0
            e = e + 1;

    pickle.dump(Scores, open(chip+'_Scores.pk', 'wb'))
    pickle.dump(X_out_I, open(chip+'_X_out_I.pk', 'wb'))
    pickle.dump(F_ids, open(chip+'_ids.pk', 'wb'))
    pickle.dump(F_periods, open(chip+'_periods.pk','wb'))

    

if __name__ == "__main__":
    main(sys.argv[1:][1]) 

