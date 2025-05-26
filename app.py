# ------------------------------
# Standard library
import sys
import copy
import importlib
import random as python_random
import time
import io
import zipfile
from functools import partial

# Basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import shap
import pickle

# Scikit-learn
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    train_test_split,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# XGBoost
import xgboost as xgb

# Custom modules
sys.path.append("..")
import utils.utils as utils 
import utils.models as models
import utils.AG_FTIR as ag
import utils.preprocessing as pp

# Initialization
shap.initjs()
importlib.reload(utils)

#Keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, TimeDistributed, Input, BatchNormalization, Bidirectional, LeakyReLU
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier

# Random seed for reproducibility
from numpy.random import seed
seed(0)
np.random.seed(123)
python_random.seed(123)
# ------------------------------

import google.protobuf
import h5py

st.title("AutoML - FTIR")
st.divider()

st.markdown("***Training and Test data***")
train_file = st.file_uploader('Training data', type=["csv"])
if train_file is not None:
    train_val_data = pd.read_csv(train_file)

    wavenumber = train_val_data.iloc[:,:-1].columns
    try:
        start_, end_ = st.select_slider(
            "Wavelength interval range to analize (Biodigital impression 1800-900)",
            options=list(wavenumber.astype(str)),
            value=(wavenumber.astype(str)[utils.get_index(train_val_data.iloc[:,:-1],1800)],
                    wavenumber.astype(str)[utils.get_index(train_val_data.iloc[:,:-1],900)]),
        )
    except:
        start_, end_ = st.select_slider(
            "Select a range of wavelength",
            options=list(wavenumber.astype(str)),
            value=(wavenumber.astype(str)[0], wavenumber.astype(str)[-1]),
        )
    
    #Range values for concatenation
    try:
        default_values = [
            wavenumber.astype(str)[utils.get_index(train_val_data.iloc[:, :-1], 3050)], #Lipidic region
            wavenumber.astype(str)[utils.get_index(train_val_data.iloc[:, :-1], 2800)]  #Lipidic region
        ]
    except Exception:
        default_values = []
        
    options = st.multiselect(
        "What ranges would you like to concatenate?",
        list(wavenumber.astype(str)),
        default=default_values,
    )

    ranges = sorted(set([start_, end_]+options), key=float, reverse=True)
    st.write("You selected:", ranges)
     
else:
    st.warning("you need to upload a csv or excel file.")

test_file = st.file_uploader('test data', type=["csv"])
if test_file is not None:
    test_data = pd.read_csv(test_file)
    # test_data


with st.sidebar:
    #Model selection
    
    st.subheader("***Models***")
    
    options = ["KNN", "SVM", "XGBoost", "BiLSTM"]
    default_selection = ["KNN"]
    model_selection = st.multiselect("Select the models:", options, default=default_selection)

    model_number = st.number_input("Insert a number for models quantitity", value=1)
    st.markdown(f"Selected models: {model_selection} x {model_number}")

    #Pre-processing
    st.divider()
    st.subheader("***Pre-processing***")
    base_options = ["Original", "PCA"]
    base_selection = st.pills("Include", base_options, selection_mode="multi", default=["Original", "PCA"])

    #AG options
    st.divider()
    st.subheader("***AG Options***")

    left, middle, right = st.columns(3, vertical_alignment="top")
    ag_generations = left.number_input("Generations", value=1)
    ag_earlystopping = middle.number_input("Early-stopping", value=1)
    ag_tp_crossover = right.number_input("Population", value=20)
    ag_n_pairs= left.number_input("X-over n_pairs", value=3)
    ag_random_prob= middle.number_input("Gen random", value=True)
    ag_cv= right.number_input("Fitness CV", value=5)

    #XAI Options
    st.divider()
    st.subheader("***Explainable Artificial Intelligence (XAI)***")

    left, right = st.columns(2, vertical_alignment="top")
    xai_knn_n_samples = left.number_input("KNN_samples", value=5)

    #Hyper-parameters
    st.divider()
    st.subheader("***Hyperparameters tunning***")
    left, right = st.columns(2, vertical_alignment="top")
    n_iter = left.number_input("n_iter", value=10)
    random_state = right.number_input('random_state', value=6)
    refit = st.selectbox(
        "Refit options",
        ("roc_auc", "accuracy", "f1_score"),
    )
    

left, middle, right= st.columns(3, vertical_alignment="top")
run = False
if (left.button("Run", type="primary", use_container_width=True)) and train_file is not None:
    run = True

loaded_model = None
if test_file is not None:
    loaded_model = right.file_uploader('Load trainned model', type=["pkl"])

if (middle.button("Run loaded Model", type="secondary", use_container_width=True)) and test_file is not None and loaded_model is not None:
    run = True

#Run Experiments
if run:
    if train_file is not None and loaded_model is None:

        train_val_classes = train_val_data.iloc[:,-1].values
        train_val_data = train_val_data.iloc[:,:-1]
        wavenumber = train_val_data.columns

        if test_file is not None:
            test_classes = test_data.iloc[:,-1].values
            test_data = test_data.iloc[:,:-1]
        else:
            st.write('We will split your training data into train/test data since there was no test data uploaded')
            train_val_data, test_data, train_val_classes, test_classes = train_test_split(
                train_val_data, train_val_classes, test_size=0.2, random_state=42
            )

        st.write('Training data:')
        train_val_data
        st.write('Test data:')
        test_data   

        st.divider()

        #Data description
        with st.spinner("Obtaining data information...", show_time=True):
            st.subheader('**Training data description**:')
            
            st.markdown('**Data plot**')
            st.pyplot(utils.plot_spectrum_streamlit([train_val_data[train_val_classes==1], train_val_data[train_val_classes==0]],label=['Positive', 'Negative'],color=['red','green'], alpha=0.1, figsize=(15,8)))
            st.markdown('**Data plot - Mean values**')
            st.pyplot(utils.plot_spectrum_streamlit([pd.DataFrame(train_val_data[train_val_classes==1].mean()).T, pd.DataFrame(train_val_data[train_val_classes==0].mean()).T],label=['Positive', 'Negative'],color=['red','green'], alpha=1, figsize=(15,8)))

            st.markdown('**Data describe**')
            st.dataframe(train_val_data.describe())

            st.markdown('**PCA 2 components**')
            st.scatter_chart(utils.visual_pca_plot_streamlit(train_val_data), x='PC1', y='PC2')

            #Test Data description
            st.subheader('**Test data description**:')
            
            st.markdown('**Data plot**')
            st.pyplot(utils.plot_spectrum_streamlit([test_data[test_classes==1], test_data[test_classes==0]],label=['Positive', 'Negative'],color=['red','green'], alpha=0.1, figsize=(15,8)))
            st.markdown('**Data plot - Mean values**')
            st.pyplot(utils.plot_spectrum_streamlit([pd.DataFrame(test_data[test_classes==1].mean()).T, pd.DataFrame(test_data[test_classes==0].mean()).T],label=['Positive', 'Negative'],color=['red','green'], alpha=1, figsize=(15,8)))

            st.markdown('**Data describe**')
            st.dataframe(test_data.describe())

            st.markdown('**PCA 2 components**')
            st.scatter_chart(utils.visual_pca_plot_streamlit(test_data), x='PC1', y='PC2')

            #Determining region intervals
            ranges = [ float(x) for x in ranges]
            pares = [(ranges[i], ranges[i+1]) for i in range(0, len(ranges)-1, 2)]
            region = [[(float(start_), float(end_))]]
            region.append(pares)
        
        #Pre-processing
        #Amida I normalization, Savitzky-Golay filter and derivations
        with st.spinner("Normalizing data...", show_time=True):
            X_normalized, X_savgol, X_savgol_df, X_savgol_df_2 = pp.apply_preprocessing(train_val_data, train_val_classes, region)
            X_test_normalized, X_test_savgol, X_test_savgol_df, X_test_savgol_df_2 = pp.apply_preprocessing(test_data, test_classes, region)

        st.badge("Normalized", icon=":material/check:", color="green")

        with st.spinner("Applying PCA on data...", show_time=True):
            # #Spectrum concatenation (when choosen more than one interval)
            ori_normalized = pp.conjugate_bases(X_normalized, X_test_normalized, 'Amida I')
            ori_savgol = pp.conjugate_bases(X_savgol, X_test_savgol, 'Savgol')
            ori_savgol_df = pp.conjugate_bases(X_savgol_df, X_test_savgol_df, 'Savgol_df')
            ori_savgol_df_2 = pp.conjugate_bases(X_savgol_df_2, X_test_savgol_df_2, 'Savgol_df')

            ori_bases = []
            if "Original" in base_selection:
                ori_bases = [ori_normalized, ori_savgol, ori_savgol_df, ori_savgol_df_2]

            #PCA application
            pca_bases = []
            if "PCA" in base_selection:
                pca_normalized = pp.pca_multi_transform(X_normalized, X_test_normalized)
                pca_savgol = pp.pca_multi_transform(X_savgol, X_test_savgol)
                pca_savgol_df = pp.pca_multi_transform(X_savgol_df, X_test_savgol_df)
                pca_savgol_df_2 = pp.pca_multi_transform(X_savgol_df_2, X_test_savgol_df_2)

                pca_bases = [pca_normalized, pca_savgol, pca_savgol_df, pca_savgol_df_2]
            

        st.badge("PCA applied", icon=":material/check:", color="green")


        #Models Selected
        xgb_model = models.xgb_model()
        knn_model = models.knn_model()
        svc_model = models.svc_model()
        bilstm_model = models.bilstm_model()

        model_dict = {
            'KNN': knn_model,
            'SVM': svc_model,
            'XGBoost': xgb_model,
            'BiLSTM': bilstm_model
        }
        #model quantitity
        train_models = []
        for i in range(model_number):
            for m in model_selection:
                train_models.append(model_dict[m])
        
        
        #Computing the best results for each model
        with st.spinner("Selecting features with Genetic algorithm...", show_time=True):
                
            best_results = []
            for model in train_models:

                pca_choosen = []
                for i in range(len(pca_bases)):
                    for base in pca_bases[i]:
                        pca_choosen.append(pp.pca_choices(xgb_model, [base], train_val_classes))

                ori_flat = []
                for i in range(len(ori_bases)):
                    for j in range(len(ori_bases[i])):
                        ori_flat.append(ori_bases[i][j])

                complete_base = ori_flat + pca_choosen

                complete_fitness = []
                for i in range(len(complete_base)):
                    pop_cal = ag.ag_pipeline(complete_base[i][0], train_val_classes, model, generations=ag_generations,Tp=ag_tp_crossover,
                                             gen_random_prob=ag_random_prob, pairs_number=ag_n_pairs, cv=ag_cv, early_stopping=ag_earlystopping)
                    complete_fitness.append(pop_cal)

                solution_base = []
                solution = [] 
                solution_fitness = 0  
                for i in range(len(complete_fitness)):
                    if complete_fitness[i][0][1] > solution_fitness:
                        solution_fitness = complete_fitness[i][0][1]
                        solution_base, solution = (complete_base[i], complete_fitness[i][0])
                
                best_results.append((model, solution_base, solution))

            # best_results = [(bilstm_model, complete_base[0], (np.random.choice(2,complete_base[0][0].shape[1]),0.5))]
        st.badge("GA applied", icon=":material/check:", color="green")

        #General configuration for the experiments
        early_stopping_rounds = 15
        n_jobs = -1
        

        #winner from model results
        # winner_fit = 0
        # winner = []
        # for results in best_results:
        #     if results[2][1] > winner_fit:
        #         winner_fit = results[2][1]
        #         winner = results

        # winner_best_model, winner_solution_base, winner_solution = winner[0], winner[1], winner[2]

        # st.write(f'Winner fitness: {solution[1]}')

    
        #Best solutions for each model
        #--------------------------------------------------------------------------------------------------------------------#
        for i_model in range(len(best_results)):

            best_model, solution_base, solution = best_results[i_model][0], best_results[i_model][1], best_results[i_model][2]

            

            st.divider()
            st.subheader('***Best model information:***')
            st.markdown(f'***Model***: {type(best_model)}')
            st.markdown(f'***Data***: {solution_base[2]} - ***Fitness***: {solution[1]}')

            mask = np.array(solution[0], dtype=bool)
            X_selected = solution_base[0].iloc[:, mask]
            test_selected = solution_base[1].iloc[:, mask]

            

            test_base = test_selected.values
            X = X_selected.values
            y = train_val_classes.astype(np.float64)

            scoring = models.scoring_params()

            k_fold = RepeatedKFold(n_splits=10, n_repeats=1, random_state=153) 

            best_model, parameters = models.get_best_model(best_model)

            if isinstance(best_model, type(xgb_model)):
                xgb_model.fit = partial(
                    xgb_model.fit,
                    early_stopping_rounds=early_stopping_rounds,
                    eval_set=[(X, y)],
                    verbose=0
                )

            if isinstance(best_model, Sequential):
                timesteps = 1
                X = X.reshape((X.shape[0], timesteps, X.shape[1]))
                test_base = test_base.reshape((test_base.shape[0], timesteps, test_base.shape[1]))

                model = models.build_bilstm_model(dropout_rate=0.3, dropout_dense=0.2, learning_rate=0.001, lstm_units=64, activation='tanh', input_shape=(timesteps, X.shape[2]))

                # Early stopping
                es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=5, restore_best_weights=True)

                best_model = KerasClassifier(
                    model=models.build_bilstm_model,
                    model__input_shape=(timesteps, X.shape[2]),
                    epochs=10000,
                    batch_size=64,
                    verbose=0,
                    callbacks=[es],
                )

            for refit in ['roc_auc']:
                
                classifier = RandomizedSearchCV(
                    estimator=best_model,
                    param_distributions=parameters,
                    n_iter=n_iter,  
                    scoring=scoring,
                    n_jobs=n_jobs,
                    cv=k_fold,
                    refit=refit,
                    return_train_score=True,
                    random_state=random_state
                )

                with st.spinner("Tunning hyperparameters...", show_time=True):
                    classifier_sv1 = classifier.fit(X, y)
                st.badge("Hyperparameters tunned", icon=":material/check:", color="green")

                # print(classifier_sv1.best_score_)
                st.markdown(f'***Hyperparameters optimization***: {classifier_sv1.best_params_}')
                st.markdown(f'***Refit***: {refit}')
                models.c_report_streamlit(classifier_sv1)
                y_pred = classifier_sv1.best_estimator_.predict(test_base)
                models.c_matrix_streamlit(true_classes=test_classes.astype(np.float64), y_pred=y_pred)

            
                

                with st.spinner("Generating model explanation...", show_time=True):
                    
                    if isinstance(best_model, type(knn_model)) or isinstance(best_model, type(svc_model)):
                        
                        X_df = pd.DataFrame(X) 
                        background = X_df.sample(n=30, random_state=42)
                        instances = pd.DataFrame(test_base[:xai_knn_n_samples, :], columns=X_selected.columns)

                        explainer = shap.KernelExplainer(classifier_sv1.predict_proba, background)
                        shap_values = explainer.shap_values(instances)
                        shap_values_transposed = np.transpose(shap_values, (2, 0, 1))

                        plt.figure()
                        shap.summary_plot(shap_values_transposed[0], features=instances, feature_names=instances.columns, show=False)
                        st.pyplot(plt.gcf())
                        plt.clf()

                    if isinstance(best_model, type(xgb_model)):
                        explainer = shap.Explainer(classifier_sv1.best_estimator_)
                        shap_values = explainer(test_base)
                        plt.figure()
                        shap.summary_plot(shap_values, test_base, feature_names=X_selected.columns)
                        st.pyplot(plt.gcf())
                        plt.clf()

                    if isinstance(best_model, KerasClassifier):
                        best_model = classifier_sv1.best_estimator_.model_
                        background = test_base[np.random.choice(test_base.shape[0], 10, replace=False)]
                        explainer = shap.GradientExplainer(best_model, background)
                        shap_values = explainer.shap_values(test_base)

                        re = np.squeeze(shap_values, axis=-1)
                        mean_shap = re.mean(axis=1)

                        plt.figure()
                        shap.summary_plot(mean_shap, features=test_base[:, 0, :], feature_names=test_selected.columns, show=False)
                        st.pyplot(plt.gcf())
                        plt.clf()

                #PCA Explanation
                try:
                    models.xai_pca_streamlit(pca=solution_base[3], columns=solution_base[4], n_contributions=solution_base[0].shape[1], n_components=10,plot=True)
                except:
                    pass

                #Model and data preparation for download
                tv_df = X_selected
                tv_df['classes'] = train_val_classes

                t_data = test_selected
                t_data['classes'] = test_classes

                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    model_buffer = io.BytesIO()
                    pickle.dump(classifier_sv1, model_buffer)
                    model_buffer.seek(0)
                    zf.writestr("model.pkl", model_buffer.read()) 

                    tv_csv = tv_df.to_csv(index=False).encode('utf-8')
                    zf.writestr("train_val_data.csv", tv_csv)

                    t_csv = t_data.to_csv(index=False).encode('utf-8')
                    zf.writestr("test_data.csv", t_csv)
                buffer.seek(0)

                st.download_button(
                    label="Save model and data",
                    data=buffer,
                    file_name="model_and_data.zip",
                    mime="application/zip"
                )

    elif test_file is not None and loaded_model is not None:

        test_classes = test_data.iloc[:,-1].values
        test_data = test_data.iloc[:,:-1]

        wavenumber = test_data.columns

        st.subheader('Test data', divider="orange")
        test_data   

        with st.spinner("Obtaining data information...", show_time=True):
            st.subheader('**Test data description**:')
            
            st.markdown('**Data plot**')
            st.pyplot(utils.plot_spectrum_streamlit([test_data[test_classes==1], test_data[test_classes==0]],label=['Positive', 'Negative'],color=['red','green'], alpha=0.1, figsize=(15,8)))
            
            st.markdown('**Data describe**')
            st.dataframe(test_data.describe())
            st.markdown(f'**Data shape**: {test_data.shape}')

        st.divider()

        

        test_base = test_data.values

        classifier_sv1 = pickle.load(loaded_model)

        if isinstance(classifier_sv1.best_estimator_, KerasClassifier):
            test_base = test_base.reshape((test_base.shape[0], 1, test_base.shape[1]))
        
        st.write(f"Model type: {type(classifier_sv1.best_estimator_)}")
        st.divider()
       
        y_pred = classifier_sv1.best_estimator_.predict(test_base)
        models.c_matrix_streamlit(true_classes=test_classes.astype(np.float64), y_pred=y_pred)

        with st.spinner("Generating model explanation...", show_time=True):
            
            if isinstance(classifier_sv1.best_estimator_, (KNeighborsClassifier, SVC)):
                
                X_df = pd.DataFrame(test_base) 
                background = X_df.sample(n=30, random_state=42)
                instances = pd.DataFrame(test_base[:xai_knn_n_samples, :], columns=wavenumber)

                explainer = shap.KernelExplainer(classifier_sv1.predict_proba, background)
                shap_values = explainer.shap_values(instances)
                shap_values_transposed = np.transpose(shap_values, (2, 0, 1))

                plt.figure()
                shap.summary_plot(shap_values_transposed[0], features=instances, feature_names=instances.columns, show=False)
                st.pyplot(plt.gcf())
                plt.clf()

            if isinstance(classifier_sv1.best_estimator_, type(xgb)):
                explainer = shap.Explainer(classifier_sv1.best_estimator_)
                shap_values = explainer(test_base)
                plt.figure()
                shap.summary_plot(shap_values, test_base, feature_names=wavenumber)
                st.pyplot(plt.gcf())
                plt.clf()

            if isinstance(classifier_sv1.best_estimator_, KerasClassifier):
                best_model = classifier_sv1.best_estimator_.model_
                background = test_base[np.random.choice(test_base.shape[0], 10, replace=False)]
                explainer = shap.GradientExplainer(best_model, background)
                shap_values = explainer.shap_values(test_base)

                re = np.squeeze(shap_values, axis=-1)
                mean_shap = re.mean(axis=1)

                plt.figure()
                shap.summary_plot(mean_shap, features=test_base[:, 0, :], feature_names=test_data.columns, show=False)
                st.pyplot(plt.gcf())
                plt.clf()


               
        #--------------------------------------------------------------------------------------------------------------------#


