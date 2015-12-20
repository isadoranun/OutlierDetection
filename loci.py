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
from sklearn.neighbors import KDTree
import statsmodels.api as sm

#import PCA stuff
from sklearn.decomposition import PCA

from sklearn.neighbors import KDTree
#LOCI implementation Adpated from Brandom Sim

# returns nhat and sigma_nhat
# nhat: average of n(p, alpha*r) over the set of r-neighbors of p
# sigma_nhat: standard deviation of n(p, alpha*r) over the set of r-neighbors of p
def nhat(p, r, alpha,points, pre_calc_n, KTree):
    
    info = {}
    #get number of neighbors of each of the neighbors
    temp = pre_calc_n[KTree.query_radius(p,  r)[0]]
    info['nhat'] = temp.mean()
    info['sigma_nhat'] = temp.std()
    if np.isnan(temp.mean() ):
        info['nhat'] = 0
        info['sigma_nhat'] = 0
    return info

# multi-granularity deviation factor
# def MDEF(e, p, r, alpha,points, pre_calc_n, KTree):

#     result = {}
#     info = nhat(p, r, alpha,points, pre_calc_n, KTree)
#     result['MDEF'] = abs(1. - (KTree.query_radius(p, r*alpha,count_only=True)) / (info['nhat']+0.00000001))
#     result['sigma_MDEF'] = info['sigma_nhat']/ (info['nhat']+0.0000000001)
#     return result

# calculates outlierliness probability
def LOCI_outliers(points, r, alpha, pre_calc_n, KTree):
    outliers = np.zeros(len(points))
    for e,p in enumerate(points):
        info = nhat(p, r, alpha,points, pre_calc_n, KTree)
#         mdef = MDEF(e,p, r, alpha,points, pre_calc_n, KTree)
#         outliers[e] =mdef['MDEF']/(mdef['sigma_MDEF'] + 0.0000000001)
        outliers[e] = (info['nhat'] - KTree.query_radius(p, r*alpha,count_only=True))/(info['sigma_nhat'] + 0.000001)
    return outliers

def loci(X_out_I, alpha=0.4):
    
    [pre_calc_n, KTree, rmax, alpha, ts_mean, ts_std, delta_out, delta_norm, thresh] = pickle.load(open('LOCI.pk', 'rb'))

        
    scaled_target=(X_out_I-ts_mean)/ (ts_std +0.0000000001)
    scores = LOCI_outliers(scaled_target, rmax, abs(alpha), pre_calc_n, KTree)
    
    scores = scores + 100

    y_scores = scores

    scores = np.zeros([len(y_scores)])
    
    for i, y_i in enumerate(y_scores):
        if y_i > thresh:
            scores[i] = 1.0/(1.0+np.exp(-1.0/delta_out*(y_i-thresh)))
        else:
            scores[i] = 1.0/(1.0+np.exp(-1.0/delta_norm*(y_i-thresh)))

    
#    scores = 1.0/(1.0+np.exp(-1.0/thresh*(scores-thresh)))
    #HIGHER IS OUTLIER
    return np.nan_to_num(scores)
