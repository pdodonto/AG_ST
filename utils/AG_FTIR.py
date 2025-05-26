from numpy.random import seed
import random as python_random
import numpy as np
import tensorflow as tf

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.model_selection import cross_val_score
import copy

from functools import partial
import xgboost as xgb

import utils.utils as utils 
import utils.models as models
import utils.preprocessing as pp

try:
    from scikeras.wrappers import KerasClassifier
    from keras.models import Sequential
    from keras.layers import Dense, Flatten, LSTM, Embedding, Dropout, TimeDistributed, Input, BatchNormalization, Bidirectional, LeakyReLU
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



def gen_initial_pop(size:int, Tp:int=10, p_1:float=0.5, random_prob:bool=False):

    pop = []

    for i in range(Tp):
        if random_prob:
            p_1 = np.random.random_sample()
        p_0 = 1 - p_1
        individual = np.random.choice([0, 1], size=size, p=[p_0, p_1])
        pop.append(individual)

    return pop


# def fitness_func(model, solution:np.array, X:pd.DataFrame, y:np.array, min_features:int=5, cv:int=5, scoring:str='accuracy'):
#     mask = np.array(solution, dtype=bool)
#     if np.sum(mask) < min_features:  #seleção mínima de features
#         return 0
#     X_selected = X.iloc[:, mask]
    
#     if isinstance(model, Sequential):
#         X_selected = X_selected.values
#         X_selected = X_selected.reshape((X_selected.shape[0], 1, X_selected.shape[1]))
#         # model = KerasClassifier(model=create_bilstm_model(1, X_selected.shape[2]), epochs=10, batch_size=32, verbose=0)

#         es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=5, restore_best_weights=True)
#         clf = KerasClassifier(
#             model=models.build_bilstm_model,
#             model__input_shape=(1, X_selected.shape[2]),
#             epochs=1000,
#             batch_size=256,
#             verbose=0,
#             callbacks=[es],
#         )
#         model = clf
    
#     scores = cross_val_score(model, X_selected, y, cv=cv, scoring=scoring)
#     return scores.mean()


# #MLP
# def create_mlp_model(input_shape):
#     model = Sequential([
#         Flatten(input_shape=input_shape),
#         Dense(64, activation='relu'),
#         Dense(1, activation='sigmoid')
#     ])
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# def fitness_func(model, solution:np.array, X:pd.DataFrame, y:np.array, min_features:int=5, cv:int=5, scoring:str='accuracy'):
#     mask = np.array(solution, dtype=bool)
#     if np.sum(mask) < min_features:  #seleção mínima de features
#         return 0
#     X_selected = X.iloc[:, mask]
    
#     if isinstance(model, Sequential):
#         X_selected = X_selected.values
#         input_shape = (X_selected.shape[1],)
#         es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=5, restore_best_weights=True)
#         clf = KerasClassifier(
#             model=create_mlp_model,
#             model__input_shape=input_shape,
#             epochs=50,
#             batch_size=32,
#             verbose=0,
#             callbacks=[es],
#         )
#         model = clf
    
#     scores = cross_val_score(model, X_selected, y, cv=cv, scoring=scoring)
#     return scores.mean()


def fitness_func(model, solution:np.array, X:pd.DataFrame, y:np.array, min_features:int=5, cv:int=5, scoring:str='accuracy'):
    mask = np.array(solution, dtype=bool)
    if np.sum(mask) < min_features:  #seleção mínima de features
        return 0
    X_selected = X.iloc[:, mask]
    
    if isinstance(model, Sequential):
        model = xgb.XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss')
    
    scores = cross_val_score(model, X_selected, y, cv=cv, scoring=scoring)
    return scores.mean()

def get_fitness(model, pop, FTIR_data:pd.DataFrame, y:np.array, cv:int=5):
    fitness = []
    for individual in pop:
        fit = fitness_func(model, individual, FTIR_data, y, cv=cv)
        fitness.append((individual, fit))
    return fitness

def order_by_fitness(pop, reverse:bool=True):
    fit_sorted = sorted(pop, key=lambda x: x[1], reverse=reverse)
    return fit_sorted




def truncation_selection(pop, p_cross:float=0.4):
    if p_cross <= 0 or p_cross > 1:
        raise Exception("p_cross values must be in interval (0,1]")
    pop_ordered = order_by_fitness(pop)
    t_p = len(pop_ordered)
    p_cross_r = round(t_p*p_cross)
    pairs_number = round(p_cross_r/2) 

    pairs = []
    for i in range(pairs_number):
        first_ind, second_ind = np.random.choice(p_cross_r, 2, replace=False) 
        pairs.append((pop_ordered[first_ind], pop_ordered[second_ind]))
    
    return pairs

def rank_based_selection(pop, pairs_number:int):

    if pairs_number <= 0:
        raise Exception("pairs_number must be positive integers")
    pop_ordered = order_by_fitness(pop, reverse=True)
    t_p = len(pop_ordered)

    ranks = np.arange(t_p, 0, -1) #linear
    p = ranks/ranks.sum()

    pairs = []
    for i in range(pairs_number):
        first_ind, second_ind = np.random.choice(t_p, 2, replace=False, p=p) #verificar
        pairs.append((pop_ordered[first_ind], pop_ordered[second_ind]))
    
    return pairs

def roulette_wheel_selection(pop, pairs_number:int, replace:bool=False):
    if pairs_number <= 0:
        raise Exception("pairs_number must be positive integers")
    
    pop_ordered = order_by_fitness(pop, reverse=True)
    t_p = len(pop_ordered)

    fitness = [individual[1] for individual in pop_ordered]
    if sum(fitness) == 0:
        raise ValueError("Cannot apply roulette wheel selection. The sum of fitness was zero")
    
    roulette_cases = fitness/sum(fitness)

    pairs = []
    for i in range(pairs_number):
        first_ind, second_ind = np.random.choice(t_p,2, replace=replace, p=roulette_cases)
        pairs.append((pop_ordered[first_ind], pop_ordered[second_ind]))
    
    return pairs


#Crossover

def simple_crossover(pairs):
    gens_number = len(pairs[0][0][0])
    if gens_number < 2:
        raise ValueError("At least 2 genes are necessary for the crossover")
                         
    new_individuals = []
    for pair in pairs:
        cut_point = np.random.randint(1,gens_number-1)
        # print(cut_point)
        first_new_born = np.concatenate((pair[0][0][:cut_point], pair[1][0][cut_point:]))
        second_new_born = np.concatenate((pair[1][0][:cut_point],pair[0][0][cut_point:]))
        new_individuals.append(first_new_born)
        new_individuals.append(second_new_born)

    return new_individuals


#Mutation

#mutation bit by bit
def binary_mutation(pop, p_mut:float=0.01):

    population = copy.deepcopy(pop)
    for individual in population:
        for i in range(len(individual)):
            mutate = np.random.choice([0, 1], size=1, p=[1-p_mut, p_mut])[0]
            if mutate == 1:
                # print("mutated")
                individual[i] = 1 - individual[i] 
    
    return population


#reinserção

def sorted_reinsertion(pop, new_pop, t_p:int=0):
    full_pop = copy.deepcopy(pop) + copy.deepcopy(new_pop)
    if t_p <= 0 or t_p > len(full_pop):
        t_p = len(pop)
    pop_ordered = order_by_fitness(full_pop, reverse=True)

    new_gen = pop_ordered[:t_p]

    return new_gen

def ag_pipeline (X, y, model, generations:int=30, Tp:int=20, gen_random_prob:bool=True, pairs_number:int=3, cv:int=5, early_stopping:int=10, insert_self:bool=False):

    FTIR_data = X
    
    initial_pop = gen_initial_pop(FTIR_data.shape[1], Tp=Tp, random_prob=gen_random_prob)
    if insert_self == True:
        initial_pop.append(np.random.randint(1,2,FTIR_data.shape[1]))    
    pop_calc = get_fitness(model, initial_pop, FTIR_data=FTIR_data, y=y, cv=cv)

    no_improves = 0
    for generation_number in range(generations):
        solution = pop_calc[0][1]
        pairs_selected = roulette_wheel_selection(pop_calc, pairs_number=pairs_number)
        new_gen = simple_crossover(pairs_selected)
        m_new_gen = binary_mutation(new_gen, p_mut=0.05)
        m_new_gen_calc = get_fitness(model, m_new_gen, FTIR_data=FTIR_data, y=y, cv=cv)
        pop_calc = sorted_reinsertion(pop_calc, m_new_gen_calc, t_p=len(pop_calc))
        if pop_calc[0][1] == solution:
            no_improves += 1
        if early_stopping == no_improves:
            break
        print(pop_calc[0][1])
    
    return pop_calc