#!/usr/bin/env python

import concatenate_results as CR
import numpy as np
import scipy as sp
import math
import csv
import copy
import glob
import pickle
import os.path
from scipy import optimize
import statsmodels.api as sm
import pickle
import pandas as pd

folder = '/n/regal/TSC/Ensemble_method3/'

[F_ids, F_periods, Scores, X_out_I] = CR.concatenate_results(folder)

print 'F_ids', F_ids
print 'X_out_I',X_out_I

[eta, mean_train, std_train] = pickle.load(open('/n/regal/TSC/Ensemble_method3/eta_F1_newDelta.pk', 'rb'))
n_exp = 5
#calculate g for each point
X_comb_I = np.nan_to_num((X_out_I - mean_train)/ (std_train+0.0000000001))

ndotx = np.dot(X_comb_I,eta)

print "el primer  minimo de ndotx  es",  np.min(ndotx)
count = 0

#for idx, row in enumerate(ndotx):
#    if np.max(row) > 709.0:
#        if np.max(row) - np.min(row) < 1418:
#		delta = (np.max(row)+np.min(row))/2
#        	ndotx[idx, :] = row - delta
#		print "I found a maximum"
#	else:
#		where = np.argmax(row)
#		if count == 0:
#			who = [idx,where]
#			count = count + 1
#		else:
#			who = np.vstack((who,[idx,where]))
#		print "NOOOOOO MAX", row
#   if np.min(row)< -709.0:
#	if np.max(row) - np.min(row) < 1418:
#                delta =	(np.max(row)+np.min(row))/2
#                ndotx[idx, :] = row - delta 
#                print "I found a minimum"
#	else:
#		where = np.argmax(row)
#               if count == 0:
#                        who = [idx,where]
#                        count = count + 1
#                else:
#                     	who = np.vstack((who,[idx,where]))
#                print "NOOOOOO MIN", row
	
        	

#ndotxmax = np.max(ndotx)
#ndotxmin = np.min(ndotx) 
#R = np.exp(ndotx- ndotxmax) 
R = np.exp(ndotx)
Q = np.sum(R, axis = 1)
Qh = np.tile(Q, (n_exp,1)).T

g = R/Qh

#print "gggg", g, "g_shape", g.shape
#print "WHO", who, "who_shape", len(who)

#if len(who) > 2:
#	for a in who:
#    		g[a[0],:] = 0.0
#    		g[a[0],a[1]] =0.0
#else:
#	g[who[0],:] = 0.0
#	g[who[0],who[1]] = 0.0

print "el minimo de ndotx  es",  np.min(ndotx)
print "el minimo de Qh es",  np.min(Qh)
print "el minimo de g es",  np.min(g)
print "el minimo de R es",  np.min(R)
if len(np.where(np.isnan(R[:,:])==True)[0])>0:
    print "Hay NAN en R" 

normalized = (g*Scores).sum(axis=1)
index = np.argsort(normalized);
index2 =index[::-1];
Outliers_ids = F_ids[index2];
Outliers_periods = F_periods[index2];
Outliers_features = X_out_I[index2,:];
Outliers_scores = normalized[index2];
g2 = g[index2,:]

#print 'diff', (ndotx- ndotxmax)[0:100,:]
#print 'ndotxmax', ndotxmax, 'max minus min', ndotxmax - ndotxmin
print 'R', R[:100,:]
print 'X_comb_I', X_comb_I[0:100,:]
print 'Scores', normalized[0:100]
print 'ndotx', ndotx[0:100,:]
print 'g', g2[0:100,:]

Outliers_ids = Outliers_ids[0:50000];
Outliers_periods = Outliers_periods[0:50000];
Outliers_features = Outliers_features[0:50000,:];
Outliers_scores = Outliers_scores[0:50000];

pickle.dump(Outliers_ids, open('IDS_1.pk', 'wb'))


pickle.dump(Scores[index2,:], open('Scores_1.pk', 'wb'))

feats = pd.DataFrame(Outliers_features)
feats.to_csv("outliersF_features_F1.csv")

resultf = open("outliersF1.csv",'wb')
wr = csv.writer(resultf, dialect='excel')

for item in Outliers_ids:
     wr.writerow([item,])

resultf = open("outliersF_periods_F1.csv",'wb')
wr = csv.writer(resultf, dialect='excel')

for item in Outliers_periods:
     wr.writerow([item,])

#resultf = open("outliersF_features_F1.csv",'wb')
#wr = csv.writer(resultf, dialect='excel')

#for item in Outliers_features:
#     wr.writerow([item,])
  
resultf = open("outliersF_scores_F1.csv",'wb')
wr = csv.writer(resultf, dialect='excel')

for item in Outliers_scores:
     wr.writerow([item,])



