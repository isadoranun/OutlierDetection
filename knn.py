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

def knn(X_out, r=0.01, n=0.9):
	
        
    [k, tree, n, d, ts_mean, ts_std,thresh_knn1, thresh_knn2, delta_out_knn1, delta_norm_knn1, delta_out_knn2, delta_norm_knn2] = pickle.load(open('KNN.pk', 'rb'))

    scaled_target= (X_out - ts_mean) / (ts_std+0.0000000001)

    distances, ind = tree.query(scaled_target, k) 

    #d = n * np.mean(distances);

    sum_distances = np.sum(distances, axis=1)

    y_scores = sum_distances
    
    sum_distances = np.zeros([len(y_scores)])
    
    for i, y_i in enumerate(y_scores):
        if y_i > thresh_knn1:
            sum_distances[i] = 1.0/(1.0+np.exp(-1.0/delta_out_knn1*(y_i-thresh_knn1)))
        else:
            sum_distances[i] = 1.0/(1.0+np.exp(-1.0/delta_norm_knn1*(y_i-thresh_knn1)))

#    sum_distances = 1.0/(1.0+np.exp(-1.0/delta_knn1*(sum_distances-thresh_knn1)))

    scores = (distances[:,:k] > d).astype('float').sum(axis=1)

    y_scores = scores
    
    scores = np.zeros([len(y_scores)])
    
    for i, y_i in enumerate(y_scores):
        if y_i > thresh_knn2:
            scores[i] = 1.0/(1.0+np.exp(-1.0/delta_out_knn2*(y_i-thresh_knn2)))
        else:
            scores[i] = 1.0/(1.0+np.exp(-1.0/delta_norm_knn2*(y_i-thresh_knn2)))


#    scores = 1.0/(1.0+np.exp(-1.0/delta_knn2*(scores-thresh_knn2)))
#     print d, distances
        
    return [sum_distances, scores]
