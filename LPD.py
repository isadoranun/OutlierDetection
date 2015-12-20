#!/usr/bin/env python

import numpy as np
import scipy as sp
import math
import csv
import copy
import glob
import pickle
import os.path

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
import statsmodels.api as sm

#import PCA stuff
from sklearn.decomposition import PCA

from sklearn.neighbors import KDTree

#modified from WESLEY CHEN
   
def LPD_outlier_score(lam, points, log_unif_p):
    
    #calculate posterior if all came from normal
    log_posterior_norm = np.log(1.-lam) + np.sum(np.nan_to_num(sp.stats.norm.logpdf(points)),axis=1)
        
    #calculate posterior if all came from uniform
    log_post_outlier = np.log(lam) - log_unif_p

    return log_post_outlier - log_posterior_norm

def LPD(X_out, lam=0.5):
     
    [lam, log_unif_p, ts_mean, ts_std, delta_out, delta_norm, thresh] = pickle.load(open('Eskin.pk', 'rb'))
    #get log_unif_p
    
    scaled_target=(X_out-ts_mean)/ (ts_std+0.0000000001)
   
    #get scores
    scores = LPD_outlier_score(lam, scaled_target, log_unif_p)
    
    y_scores = scores
    
    scores = np.zeros([len(y_scores)])
    
    for i, y_i in enumerate(y_scores):
        if y_i > thresh:
            scores[i] = 1.0/(1.0+np.exp(-1.0/delta_out*(y_i-thresh)))
        else:
            scores[i] = 1.0/(1.0+np.exp(-1.0/delta_norm*(y_i-thresh)))

#    scores = 1.0/(1.0+np.exp(-1.0/delta*(scores-thresh)))
    

    #HIGHER IS OUTLIER
    return scores
