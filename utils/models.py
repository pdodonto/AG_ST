# ------------------------------
# Standard Library
import random as python_random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import seed
from scipy.stats import uniform, randint

# Machine Learning
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import xgboost as xgb

# Deep Learning (TensorFlow / Keras)
import tensorflow as tf

try:
    from keras.models import Sequential
    from keras.layers import (
        Dense,
        LSTM,
        Embedding,
        Dropout,
        TimeDistributed,
        Input,
        BatchNormalization,
        Bidirectional,
        LeakyReLU,
    )
    from keras.preprocessing import sequence
    from keras.callbacks import EarlyStopping
    from keras.optimizers import Adam
except ImportError:
    pass

# Web Interface
import streamlit as st

# Random seed for reproducibility
seed(0)
np.random.seed(123)
python_random.seed(123)

try:
    tf.random.set_seed(7)
except Exception:
    pass
# ------------------------------


# Função para calcular especificidade
def specificity_score(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def f1_modified (y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    recall = tp / (tp + fn)
    f1 = 2/(1/specificity+1/recall)
    return f1

def scoring_params ():

    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='binary'),
        'recall': make_scorer(recall_score, average='binary'),
        'specificity': make_scorer(specificity_score),
        'f1_score': make_scorer(f1_modified),
        'roc_auc': 'roc_auc'
    }
    return scoring

def c_report_streamlit(classifier_sv1):

    best_index = classifier_sv1.best_index_
    cv = classifier_sv1.cv_results_

    
    st.subheader("\n**:orange[Classification Report (CV) Train-Validation]**\n", divider='orange')

    st.markdown("**AUC-ROC**: {:.3f} ±{:.3f}".format(
        cv['mean_test_roc_auc'][best_index],
        cv['std_test_roc_auc'][best_index]
    ))
    st.markdown("**Accuracy**: {:.3f} ±{:.3f}".format(
        cv['mean_test_accuracy'][best_index],
        cv['std_test_accuracy'][best_index]
    ))
    st.markdown("**Precision**: {:.3f} ±{:.3f}".format(
        cv['mean_test_precision'][best_index],
        cv['std_test_precision'][best_index]
    ))
    st.markdown("**Specificity**: {:.3f} ±{:.3f}".format(
        cv['mean_test_specificity'][best_index],
        cv['std_test_specificity'][best_index]
    ))
    st.markdown("**Sensibility**: {:.3f} ±{:.3f}".format(
        cv['mean_test_recall'][best_index],
        cv['std_test_recall'][best_index]
    ))
    st.markdown("**F1-score**: {:.3f} ±{:.3f}".format(
        cv['mean_test_f1_score'][best_index],
        cv['std_test_f1_score'][best_index], end=''
    ))
    st.subheader('', divider='orange')

def c_report(classifier_sv1):

    best_index = classifier_sv1.best_index_
    cv = classifier_sv1.cv_results_
    print("\n--- Classification Report (CV) Train-Validation ---\n")
    print("AUC-ROC: {:.3f} ±{:.3f}".format(
        cv['mean_test_roc_auc'][best_index],
        cv['std_test_roc_auc'][best_index]
    ))
    print("Acurácia: {:.3f} ±{:.3f}".format(
        cv['mean_test_accuracy'][best_index],
        cv['std_test_accuracy'][best_index]
    ))
    print("Precisão: {:.3f} ±{:.3f}".format(
        cv['mean_test_precision'][best_index],
        cv['std_test_precision'][best_index]
    ))
    print("Especificidade: {:.3f} ±{:.3f}".format(
        cv['mean_test_specificity'][best_index],
        cv['std_test_specificity'][best_index]
    ))
    print("Sensibilidade: {:.3f} ±{:.3f}".format(
        cv['mean_test_recall'][best_index],
        cv['std_test_recall'][best_index]
    ))
    print("F1-score: {:.3f} ±{:.3f}".format(
        cv['mean_test_f1_score'][best_index],
        cv['std_test_f1_score'][best_index]
    ))
    print("\n---------------------------------------------------\n")



def c_matrix_streamlit(true_classes, y_pred):


    st.subheader('Classification Report (Test Set)', divider='gray')
  
    cm = confusion_matrix(true_classes, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    # Classification report table
    report_dict = classification_report(true_classes, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.round(2))

    # Highlighted summary metrics
    st.subheader('Metrics Summary')

    sensitivity = report_dict['1.0']['recall']
    specificity = report_dict['0.0']['recall']
    accuracy = report_dict['accuracy']
    precision = report_dict['1.0']['precision']
    f1_custom = 2 * (sensitivity * specificity) / (sensitivity + specificity + 1e-6)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2f}")
        st.metric("Precision (Class 1)", f"{precision:.2f}")
        st.metric("(Sensitivity-Specificity)\n F1-Score", f"{f1_custom:.2f}")
    with col2:
        st.metric("Sensitivity (Recall Class 1)", f"{sensitivity:.2f}")
        st.metric("Specificity (Recall Class 0)", f"{specificity:.2f}")


def c_matrix(true_classes, y_pred):
    print('\n------------------ Confusion Matrix -----------------\n')
    print(confusion_matrix(true_classes, y_pred))

    print('\n----------- Classification Report - Test ------------')
    print(classification_report(true_classes, y_pred))
    print('----------------------- Model -----------------------')

    report = classification_report(true_classes, y_pred, output_dict=True)
    sensibilidade = report['1.0']['recall']
    especificidade = report['0.0']['recall']
    acuracia = report['accuracy']
    precisao = report['1.0']['precision']
    fscore = 2*(sensibilidade*especificidade)/(sensibilidade+especificidade)

    print('Acurácia: {:.2f}'.format(acuracia))
    print('Precisão: {:.2f}'.format(precisao))
    print('Sensibilidade: {:.2f}'.format(sensibilidade))
    print('Especificidade: {:.2f}'.format(especificidade))
    print('F1-score: {:.2f}'.format(fscore))

def std_xgb_param ():
    parameters = {  
        'objective': ['binary:logistic'],
        'n_estimators': [50,75,100],
        'eta': np.concatenate((np.arange(0.1, 1, 0.001), [0.7965]), axis=0),
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.65, 0.8, 1.0],
        'seed': [6],
        'max_depth': [3,4,5,6],
        'colsample_bytree': [0.6, 0.8, 1.0],             
        'min_child_weight': [1, 3, 5, 10],               
        'gamma': [0, 0.1, 0.3],                          
        'reg_alpha': [0, 0.1, 1.0],                     
        'reg_lambda': [0.5, 1.0, 2.0]    
    }
    return parameters

def std_knn_param():
    parameters = {
        'n_neighbors': [1,3,5,7,9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'cosine']
    }
    return parameters

def std_svc_param():
    parameters = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'random_state': [2]
    }
    return parameters

def std_bilstm_param():
    parameters = {
        'model__dropout_rate': np.arange(0.1, 0.9, 0.1),
        # 'model__learning_rate': [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        'model__learning_rate': [ 0.001, 0.005],
        'model__lstm_units': [64],
        # 'model__lstm_units': [64, 128, 256],
        'batch_size': [64],
        # 'batch_size': [32, 64],
        'model__dropout_dense': np.arange(0.1, 0.7, 0.1),
        'model__activation': ['tanh', 'relu']
    }
    return parameters

def rs_cv_params():
    param_dist = {
        'objective': ['binary:logistic'],
        'n_estimators': [50,100],  
        'eta': uniform(0.01, 0.95),  
        'subsample': [0.65],  
        'seed': [6],         
        'max_depth': randint(2, 6), 
        'reg_lambda': uniform(0.1, 1.0)  
    }
    return param_dist




def build_bilstm_model(dropout_rate=0.3, dropout_dense=0.2, learning_rate=0.0001, lstm_units=256, activation='relu', input_shape=None):
    model = Sequential()
    model.add(Input(input_shape))
    model.add(Bidirectional(LSTM(lstm_units, dropout=dropout_rate, activation=activation)))
    model.add(Dropout(dropout_dense))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def xai_pca(pca, columns, n_contributions=5, n_components=10, plot=False):

    loadings = pd.DataFrame(
        pca.components_.T,  # shape: (n_original_features, n_components)
        index=columns,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )

    for pc in loadings.columns:
        print(f'\n{pc} - top {n_contributions} contribuições:')
        print(loadings[pc].abs().sort_values(ascending=False).head(n_contributions))
    
    if n_components > loadings.shape[1]:
        n_components = loadings.shape[1]

    if plot == True:
        plt.figure(figsize=(12, 8))
        sns.heatmap(loadings.iloc[:, :n_components], cmap="coolwarm", center=0)
        plt.title("PCA Loadings: contribuição das variáveis para cada componente")
        plt.xlabel("Componentes Principais")
        plt.ylabel("Variáveis Originais")
        plt.show()

# def xai_pca_streamlit(pca, columns, n_contributions=5, n_components=10, plot=False):

#     loadings = pd.DataFrame(
#         pca.components_.T,
#         index=columns,
#         columns=[f"PC{i+1}" for i in range(pca.n_components_)]
#     )

#     for pc in loadings.columns:
#         st.write(f'**{pc} - top {n_contributions} contribuições:**')
#         top_vars = loadings[pc].abs().sort_values(ascending=False).head(n_contributions)
#         st.dataframe(top_vars)

#     if n_components > loadings.shape[1]:
#         n_components = loadings.shape[1]

#     # Exibe o heatmap se plot=True
#     if plot:
#         fig, ax = plt.subplots(figsize=(12, 8))
#         sns.heatmap(loadings.iloc[:, :n_components], cmap="coolwarm", center=0, ax=ax)
#         ax.set_title("PCA Loadings: contribuição das variáveis para cada componente")
#         ax.set_xlabel("Componentes Principais")
#         ax.set_ylabel("Variáveis Originais")
#         st.pyplot(fig)


def xai_pca_streamlit(pca, columns, n_contributions=5, n_components=10, plot=False):
    loadings = pd.DataFrame(
        pca.components_.T,
        index=columns,
        columns=[f"PC{i+1} - CVR: {sum(pca.explained_variance_ratio_[:i+1])*100:.5f}%" for i in range(pca.n_components_)]
    )

    st.markdown("**Wavenumber contributions per component**")
    expander = st.expander("PCA explanation")
    with expander.container(height=300):
        for pc in loadings.columns:
            top_vars = loadings[pc].abs().sort_values(ascending=False).head(n_contributions)
            df_display = pd.DataFrame({
                "Wavenumber": top_vars.index,
                "Contribution": top_vars.values
            })
            st.markdown(f"**{pc}**")
            st.dataframe(df_display.round({"Contribution": 4}), height=200)

    if n_components > loadings.shape[1]:
        n_components = loadings.shape[1]

    if plot:
        st.markdown("---")
        st.markdown("**Contributions Heatmap**")
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(loadings.iloc[:, :n_components], cmap="coolwarm", center=0, ax=ax,
                    cbar_kws={'label': 'Value'})
        ax.set_title(f"""PCA first {n_components} out of {pca.n_components_} components.
                    Cumulative Variance Ratio (CVR) of {sum(pca.explained_variance_ratio_[:n_components])*100:.5f}% out of {sum(pca.explained_variance_ratio_)*100:.2f}%.""")
        ax.set_xlabel("Most contribution")
        ax.set_ylabel("Original")
        st.pyplot(fig)

def xgb_model():
    model = xgb.XGBClassifier(objective='binary:logistic', 
                            n_estimators= 50,
                            # early_stopping_rounds=10,
                            seed=6,
                            subsample=0.65, 
                            max_depth=3, 
                            eta=0.4,
                            reg_lambda=0.6,)
    return model

def knn_model(n_neighbors:int=5):
    model =  KNeighborsClassifier(n_neighbors)
    return model

def svc_model():
    model = SVC(probability=True)
    return model

def bilstm_model():
    model = Sequential()
    return model


def get_best_model(best):
    
    if isinstance(best, type(knn_model())):
        model = knn_model()
        parameters = std_knn_param()
    
    elif isinstance(best, type(xgb.XGBClassifier())):
        model = xgb.XGBClassifier()
        # Ajuste o método fit para sempre incluir early_stopping_rounds, eval_set e verbose
        parameters = std_xgb_param()

    elif isinstance(best, type(svc_model())):
        model = svc_model()
        parameters = std_svc_param()
    
    else:
        model = Sequential()
        parameters = std_bilstm_param()

    return model, parameters

