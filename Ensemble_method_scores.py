%matplotlib inline
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
plt.rcParams['savefig.dpi'] = 300 
import seaborn as sns
import csv
import copy
import glob
import pickle
import os.path
import sklearn

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
from statsmodels.robust.scale import mad as MAD