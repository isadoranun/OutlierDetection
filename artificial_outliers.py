import numpy as np

#creates artificial outliers
def artificial_outliers(X, factor, out_n=10):
    
    indices = np.random.permutation(len(X))[:out_n]
#   
    X_artifical_outliers = X[indices]

    #randomize continuous features
    for temp in range(0,100): #shuffle 100 times
        for cf in range(0,len(X[0])): #iterate over continuous features
            np.random.shuffle(X_artifical_outliers[:,cf])

    for cf in range(0,len(X[0])): #iterate over continuous features, and add standard deviations away from mean
        X_artifical_outliers[:,cf] = X_artifical_outliers[:,cf]+(np.random.randn(out_n))*factor*X[:,cf].std()

    return X_artifical_outliers