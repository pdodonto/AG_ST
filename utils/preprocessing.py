from numpy.random import seed
import random as python_random
import numpy as np
import tensorflow as tf

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

import sys
sys.path.append("..")

import importlib
import utils.utils as utils 
import utils.models as models
import utils.AG_FTIR as ag


try:
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, Embedding, Dropout, TimeDistributed, Input, BatchNormalization, Bidirectional, LeakyReLU
    from keras.preprocessing import sequence
    from keras.callbacks import EarlyStopping
    from keras.optimizers import Adam
except:
    pass

seed(0)
try:
    tf.random.set_seed(7)
except:
    pass
    
np.random.seed(123)
python_random.seed(123)



from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis

# def detect_outliers_mahalanobis(X, n_components=10, threshold=3):
#     pca = PCA(n_components=n_components)
#     X_pca = pca.fit_transform(X)
#     mean = X_pca.mean(axis=0)
#     cov = np.cov(X_pca.T)
#     cov_inv = np.linalg.inv(cov)
    
#     distances = np.array([
#         mahalanobis(x, mean, cov_inv) for x in X_pca
#     ])
#     mask = distances < threshold
#     return X[mask], mask

# def plot_boxplot_features(X, feature_names):
#     plt.figure(figsize=(20,5))
#     sns.boxplot(data=pd.DataFrame(X, columns=feature_names))
#     plt.xticks(rotation=90)
#     plt.title("Boxplot das bandas espectrais")
#     plt.show()

# from sklearn.ensemble import IsolationForest

# def detect_outliers_isolation_forest(X, contamination=0.05):
#     iso = IsolationForest(contamination=contamination, random_state=42)
#     y_pred = iso.fit_predict(X)
#     mask = y_pred == 1  # 1 = inlier, -1 = outlier
#     return X[mask], mask

# def apply_preprocessing(X,y,region=[[(1800,900)]]): #region=[[(1800,900)],[(3050,2800),(1800,900)]]
def apply_preprocessing(X,y,region=[[(1800,900)],[(3050,2800),(1800,900)]]): #
    
    bases = []
    
    for interval in region:
        base = []
        for i in range(len(interval)):
            wave1 = utils.get_index(X,interval[i][0])
            wave2 = utils.get_index(X,interval[i][1])
            base.append(X.iloc[:,wave1:wave2])
        bases.append(pd.concat((seg for seg in base), axis=1))

    #Baseline correction ---> implement later

    #Amida I normalization
    normalized = []

    #Savitzky-golay
    polyorder = 4
    window_length = 20
    savgol = []
    savgol_df = []
    savgol_df_2 = []

    for i in range(len(bases)):
        s0 = utils.apply_savgol(bases[i], window_length=window_length, polyorder=polyorder, deriv=0)
        s1 = utils.apply_savgol(bases[i], window_length=window_length, polyorder=polyorder, deriv=1)
        s2 = utils.apply_savgol(bases[i], window_length=window_length, polyorder=polyorder, deriv=2)

        a1_valid = False
        for j in range(len(region)):
            for tuples in region[j]:
                if tuples[0] >= 1700 and tuples[1] <= 1600:
                    a1_valid=True

        if a1_valid:
            columns = bases[i].columns.values.astype(np.float64)
            # print(len(columns))
            normalized.append(utils.normalize_amida_I(bases[i], columns))
            #Savitzky-golay filter
            savgol.append(utils.normalize_amida_I(s0, columns))
            #Savitzky-golay filter + first derivative
            savgol_df.append(utils.normalize_amida_I(s1, columns))
            #Savitzky-golay filter + second derivative
            savgol_df_2.append(utils.normalize_amida_I(s2, columns))
        else:
            savgol.append(s0)
            savgol_df.append(s1)
            savgol_df_2.append(s2)
        
    return normalized, savgol, savgol_df, savgol_df_2


def pca_transform(FTIR_data, test_data, num_components):
    pca = PCA(n_components=num_components)
    x_t = pd.DataFrame(pca.fit_transform(FTIR_data))
    t_t = pca.transform(test_data)

    return x_t, t_t, pca


def pca_multi_transform (X_bases, X_test_bases, vars=[0.95, 0.99, 0.999, 0.9999]):

    pca_data = []
    for i in range(len(X_bases)):
        pca_var = []
        for var in vars:  
            x_t, t_t, pca = pca_transform(X_bases[i], X_test_bases[i], var)
            pca_var.append((pd.DataFrame(x_t), pd.DataFrame(t_t), f'PCA({i}): {var} - init_cols({X_bases[i].shape[1]})', pca, X_test_bases[i].columns))
        pca_data.append(pca_var)

    return pca_data

    #pca_data[0] --> first base
    #pca_data[0][0] ---> first var
    #pca_data[0][0][i] ----> i==0 X, i==1 test, i==2 var


#implementar MNF


def conjugate_bases(a, b, value):
    base = []
    for i in range(len(a)):
        base.append((a[i],b[i],f'{value}: ({a[i].columns[0]} - {a[i].columns[-1]})'))
    return base


def pca_choices(model, bases, y):
    pca_choosen = []
    fitness = 0
    for base in bases:
        for i in range(len(base)):
            train = base[i][0]
            solution = np.random.randint(1,2,train.shape[1])
            valor = ag.fitness_func(model, solution, train, y)
            if valor > fitness:
                fitness = valor
                pca_choosen = base[i] 
        
            # print(f'PCA:{base[i][2]}, {valor}')
            # print(fitness)
    return pca_choosen