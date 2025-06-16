# ------------------------------
# Standard Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import uniform, randint

# Machine Learning
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import xgboost as xgb

from sklearn.model_selection import (
    RepeatedKFold,
    RandomizedSearchCV
)

# Deep Learning (TensorFlow / Keras)
import tensorflow as tf

from keras.models import Sequential
from keras.layers import (
    Dense,
    LSTM,
    Dropout,
    Input,
    Bidirectional,
)

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasClassifier

# Web Interface
import streamlit as st
from functools import partial

# ------------------------------


# Função para calcular especificidade
def specificity_score(y_true, y_pred):
    """
    Calcula o valor da especificidade (specificity) para problemas de classificação binária.

    A especificidade é a proporção de verdadeiros negativos corretamente identificados,
    também conhecida como taxa de verdadeiros negativos (True Negative Rate).

    Parâmetros
    ----------
    y_true : array-like
        Vetor contendo os rótulos reais das classes (0 ou 1).
    y_pred : array-like
        Vetor contendo os rótulos preditos pelo modelo (0 ou 1).

    Retorno
    -------
    specificity : float
        Valor da especificidade, variando de 0 a 1. Calculado como: TN / (TN + FP)

    Observações
    -----------
    - Requer que as classes estejam codificadas como 0 (negativo) e 1 (positivo).
    - Caso não haja verdadeiros negativos ou falsos positivos, pode ocorrer divisão por zero.

    Exemplo
    -------
    >>> y_true = [0, 1, 0, 1, 0, 0, 1]
    >>> y_pred = [0, 1, 1, 1, 0, 0, 0]
    >>> spec = specificity_score(y_true, y_pred)
    >>> print(f"Specificity: {spec:.2f}")
    Specificity: 0.75
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def f1_modified (y_true, y_pred):
    """
    Calcula uma métrica F1 modificada, baseada na média harmônica entre a especificidade (specificity) e o recall (sensibilidade).

    Esta métrica é útil para problemas de classificação binária onde é importante balancear o desempenho entre 
    verdadeiros negativos (especificidade) e verdadeiros positivos (recall/sensibilidade), especialmente em contextos biomédicos.

    Parâmetros
    ----------
    y_true : array-like
        Vetor com os rótulos verdadeiros das classes (0 para negativo, 1 para positivo).
    y_pred : array-like
        Vetor com os rótulos previstos pelo modelo (0 ou 1).

    Retorno
    -------
    f1 : float
        Valor da F1 modificada, variando de 0 a 1. Retorna 0 caso recall ou especificidade sejam 0.

    Observações
    -----------
    - A especificidade é calculada como TN / (TN + FP).
    - O recall (sensibilidade) é calculado como TP / (TP + FN).
    - A F1 modificada é a média harmônica entre especificidade e recall:
        F1 = 2 / (1/specificity + 1/recall)
    - Caso recall ou especificidade sejam 0, a função retorna 0 para evitar divisão por zero.

    Exemplo
    -------
    >>> y_true = [0, 1, 0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 1, 1, 0, 0, 1]
    >>> f1_mod = f1_modified(y_true, y_pred)
    >>> print(f"F1 Modificada: {f1_mod:.2f}")
    F1 Modificada: 0.80
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    recall = tp / (tp + fn)
    if specificity == 0 or recall == 0:
        f1 = 0 
    else:
        f1 = 2 / (1/specificity + 1/recall)
    return f1

def scoring_params ():
    """
    Retorna um dicionário de métricas de avaliação (scoring) para uso em validação de modelos em machine learning.

    Este dicionário é apropriado para utilização em funções do Scikit-learn como `GridSearchCV` e `cross_validate`, 
    permitindo a avaliação do desempenho dos modelos com múltiplas métricas, incluindo métricas personalizadas 
    como especificidade e uma F1 modificada.

    Parâmetros
    ----------
    Nenhum.

    Retorno
    -------
    scoring : dict
        Dicionário contendo as seguintes métricas:
            - 'accuracy': Acurácia (proporção de previsões corretas).
            - 'precision': Precisão para a classe positiva (proporção de positivos previstos que são verdadeiros).
            - 'recall': Sensibilidade (proporção de positivos corretamente identificados).
            - 'specificity': Especificidade (proporção de negativos corretamente identificados, usando função customizada).
            - 'f1_score': F1 Score modificado, baseado na média harmônica entre especificidade e recall.
            - 'roc_auc': Área sob a curva ROC.

    Observações
    -----------
    - As métricas 'precision' e 'recall' são configuradas para classificação binária.
    - 'specificity' e 'f1_score' são calculados usando funções definidas pelo usuário.
    - O dicionário pode ser passado diretamente para os parâmetros `scoring` de funções do Scikit-learn.

    Exemplo
    -------
    >>> from sklearn.model_selection import cross_validate
    >>> scores = cross_validate(model, X, y, cv=5, scoring=scoring_params())
    >>> print(scores['test_specificity'])
    [0.80 0.90 0.85 0.88 0.87]
    """
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
    """
    Exibe, no Streamlit, um relatório de métricas de classificação para o melhor modelo obtido via validação cruzada.

    A função extrai e apresenta, de forma formatada e amigável, as principais métricas de avaliação 
    (AUC-ROC, acurácia, precisão, especificidade, sensibilidade e F1-score) do melhor modelo encontrado em uma busca 
    de hiperparâmetros (por exemplo, GridSearchCV ou RandomizedSearchCV).

    Parâmetros
    ----------
    classifier_sv1 : sklearn.model_selection.GridSearchCV ou RandomizedSearchCV
        Objeto de busca de hiperparâmetros já ajustado (fit), contendo os resultados de validação cruzada 
        no atributo `cv_results_` e o índice do melhor modelo em `best_index_`.

    Retorno
    -------
    None
        Esta função não retorna valores; ela exibe as métricas diretamente na interface do Streamlit.

    Observações
    -----------
    - As métricas exibidas são apresentadas no formato "média ± desvio padrão" sobre os folds de validação cruzada.
    - É necessário que as métricas customizadas (como 'specificity' e 'f1_score') tenham sido definidas no parâmetro `scoring` ao ajustar o `classifier_sv1`.
    - Utiliza marcação e divisores coloridos para melhorar a visualização no Streamlit.

    Exemplo
    -------
    >>> from sklearn.model_selection import GridSearchCV
    >>> # Supondo que classifier_sv1 já foi ajustado e possui as métricas configuradas
    >>> c_report_streamlit(classifier_sv1)
    # Exibe relatório na interface Streamlit

    """
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
    """
    Exibe a matriz de confusão e um relatório de métricas de classificação para o conjunto de teste no Streamlit.

    Esta função apresenta uma visualização gráfica da matriz de confusão, uma tabela detalhada do relatório de classificação
    e um resumo das principais métricas de desempenho, facilitando a interpretação dos resultados de um classificador.

    Parâmetros
    ----------
    true_classes : array-like
        Vetor contendo os rótulos reais das classes do conjunto de teste.
    y_pred : array-like
        Vetor contendo os rótulos previstos pelo modelo para o conjunto de teste.

    Retorno
    -------
    None
        A função não retorna valores; exibe as visualizações e métricas diretamente na interface do Streamlit.

    Observações
    -----------
    - Exibe a matriz de confusão como um heatmap com valores anotados.
    - Mostra um relatório detalhado de classificação (precision, recall, f1-score, support) em formato de tabela.
    - Destaca as principais métricas: acurácia, precisão, sensibilidade (recall da classe 1), especificidade (recall da classe 0)
      e uma F1 customizada baseada na média harmônica entre sensibilidade e especificidade.
    - Utiliza colunas no Streamlit para apresentação visual organizada dos resultados.

    Exemplo
    -------
    >>> y_true = [0, 1, 0, 1, 1, 0]
    >>> y_pred = [0, 1, 0, 0, 1, 1]
    >>> c_matrix_streamlit(y_true, y_pred)
    # Exibe matriz de confusão e métricas no Streamlit

    """
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

def std_xgb_param():
    """
    Retorna um dicionário de hiperparâmetros padrão para busca em Grid Search no modelo XGBoost (classificação binária).

    Os hiperparâmetros abrangem número de árvores, profundidade, taxa de aprendizado, regularização e parâmetros de ensemble, cobrindo amplo espaço de busca.

    Parâmetros
    ----------
    Nenhum.

    Retorno
    -------
    parameters : dict
        Dicionário com listas de valores para os principais hiperparâmetros do XGBoost.

    Exemplo
    -------
    >>> from sklearn.model_selection import GridSearchCV
    >>> params = std_xgb_param()
    >>> grid = GridSearchCV(xgb.XGBClassifier(), params, cv=5)
    """
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
    """
    Retorna um dicionário de hiperparâmetros padrão para busca em Grid Search no modelo KNeighborsClassifier.

    """
    parameters = {
        'n_neighbors': [1,3,5,7,9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'cosine']
    }
    return parameters

def std_lda_param():
    """
    Retorna um dicionário de hiperparâmetros padrão para busca em Grid Search no modelo Linear Discriminant Analysis (LDA).

    """
    parameters = {
        'solver': ['svd'],
    }
    return parameters

def std_svc_param():
    """
    Retorna um dicionário de hiperparâmetros padrão para busca em Grid Search no modelo Support Vector Classifier (SVC).

    """
    parameters = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        # 'random_state': [2]
    }
    return parameters

def std_bilstm_param():
    """
    Retorna um dicionário de hiperparâmetros padrão para busca em Grid Search de um modelo BiLSTM (Keras/SciKeras).

    """
    parameters = {
        'model__dropout_rate': np.arange(0.1, 0.9, 0.1),
        'model__learning_rate': [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        'model__lstm_units': [64, 128, 256],
        'batch_size': [16,32,64],
        'model__dropout_dense': np.arange(0.1, 0.7, 0.1),
        'model__activation': ['tanh', 'relu']
    }
    return parameters

def rs_cv_params():
    """
    Retorna um dicionário de hiperparâmetros para busca aleatória (RandomizedSearchCV) no modelo XGBoost.

    """
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
    """
    Constrói e retorna um modelo Bidirectional LSTM (BiLSTM) para classificação binária usando Keras/TensorFlow.

    O modelo é composto por uma camada LSTM bidirecional com dropout, uma camada densa de dropout adicional,
    e uma camada de saída densa com ativação sigmoide. O otimizador é Adam e a função de perda é 'binary_crossentropy'.

    Parâmetros
    ----------
    dropout_rate : float, opcional (default=0.3)
        Taxa de dropout aplicada na camada LSTM para regularização.
    dropout_dense : float, opcional (default=0.2)
        Taxa de dropout aplicada antes da camada de saída (Dense).
    learning_rate : float, opcional (default=0.0001)
        Taxa de aprendizado do otimizador Adam.
    lstm_units : int, opcional (default=256)
        Número de unidades (neurônios) na camada LSTM bidirecional.
    activation : str, opcional (default='relu')
        Função de ativação usada nas células LSTM.
    input_shape : tuple, obrigatório
        Shape da entrada, no formato (timesteps, features).

    Retorno
    -------
    model : keras.models.Sequential
        Modelo Keras Sequential pronto para ser treinado com métodos como `fit()`.

    Observações
    -----------
    - O modelo é projetado para tarefas de classificação binária.
    - Ideal para aplicações em séries temporais, espectros FTIR ou sinais biomédicos multivariados.
    - Necessita do TensorFlow/Keras instalado.

    Exemplo
    -------
    >>> model = build_bilstm_model(input_shape=(10, 20))
    >>> model.summary()
    """
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


def xai_pca_streamlit(pca, columns, metadata="", n_contributions=5, n_components=10, plot=False):
    """
    Exibe análise explicativa de PCA (Principal Component Analysis) para espectroscopia ou dados multivariados em Streamlit,
    destacando as principais contribuições de cada componente e (opcionalmente) um heatmap das contribuições das features.

    Parâmetros
    ----------
    pca : sklearn.decomposition.PCA
        Objeto PCA já ajustado, contendo os componentes principais e variância explicada.
    columns : list ou array-like
        Lista dos nomes das variáveis originais (ex: números de onda, atributos).
    metadata : str, opcional (default="")
        Texto adicional para identificar/explicar a análise, mostrado no título do expander.
    n_contributions : int, opcional (default=5)
        Número de maiores contribuições (features mais relevantes) a serem exibidas por componente principal.
    n_components : int, opcional (default=10)
        Número de componentes principais a serem exibidos/analisados.
    plot : bool, opcional (default=False)
        Se True, gera e exibe um heatmap das contribuições dos atributos para os componentes selecionados.

    Retorno
    -------
    loadings : pd.DataFrame
        DataFrame contendo os loadings (contribuições) de cada feature para cada componente principal.
    heatmap_buffer : BytesIO or None
        Buffer PNG do heatmap gerado, ou None caso `plot` seja False.

    Observações
    -----------
    - Exibe, em Streamlit, as contribuições de cada feature para os componentes principais, facilitando interpretação e explicabilidade (XAI).
    - O heatmap é salvo em buffer para possível download ou uso posterior.
    - O texto `metadata` pode ser usado para contextualizar diferentes análises (ex: nome do dataset, etapa do pipeline, etc.).

    Exemplo
    -------
    >>> loadings, heatmap_buf = xai_pca_streamlit(pca, columns, n_contributions=7, plot=True)
    # Exibe tabelas e heatmap em Streamlit; loadings pode ser exportado para análise adicional.

    """
    import io

    loadings = pd.DataFrame(
        pca.components_.T,
        index=columns,
        columns=[f"PC{i} - CVR: {sum(pca.explained_variance_ratio_[:i])*100:.5f}%" for i in range(pca.n_components_)]
    )

    st.markdown("**Wavenumber contributions per component**")
    expander = st.expander(f"PCA explanation {metadata}")
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

    heatmap_buffer = None
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

        # Salva o heatmap em buffer PNG
        heatmap_buffer = io.BytesIO()
        fig.savefig(heatmap_buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        heatmap_buffer.seek(0)

    return loadings, heatmap_buffer

def xgb_model():
    """
    Cria e retorna um modelo XGBoost para classificação binária com parâmetros pré-definidos.
    """
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
    """
    Cria e retorna um modelo KNN para classificação binária com parâmetros pré-definidos.
    """
    model =  KNeighborsClassifier(n_neighbors)
    return model

def lda_model(solver:str='svd'):
    """
    Cria e retorna um modelo LDA para classificação binária com parâmetros pré-definidos.
    """
    model = LinearDiscriminantAnalysis(solver)
    return model

def svc_model():
    """
    Cria e retorna um modelo SVM para classificação binária com parâmetros pré-definidos.
    """
    model = SVC(probability=True, max_iter=100000)
    return model

def bilstm_model():
    """
    Cria e retorna um modelo sequencial Keras vazio, para posterior configuração de uma rede BiLSTM.
    """
    model = Sequential()
    return model


def get_best_model(best):
    """
    Identifica o tipo de modelo fornecido e retorna uma nova instância desse modelo junto com o dicionário padrão de hiperparâmetros.

    Esta função é útil para pipelines automatizados (AutoML) ou buscas evolutivas, permitindo instanciar e buscar hiperparâmetros
    apropriados conforme o tipo do melhor modelo identificado previamente.

    Parâmetros
    ----------
    best : object
        Modelo (estimador) já treinado ou identificado como "melhor" por algum método de seleção, como GridSearchCV, RandomizedSearchCV, ou Algoritmo Genético.

    Retorno
    -------
    model : estimador
        Nova instância do modelo identificado (KNN, LDA, XGBoost, SVC, ou BiLSTM).
    parameters : dict
        Dicionário de hiperparâmetros padrão para busca ou ajuste do modelo correspondente.

    Observações
    -----------
    - A identificação é feita por comparação de tipos usando instâncias vazias dos modelos.
    - Para modelos KNN, LDA, XGBoost e SVC, utiliza as funções auxiliares de construção e dicionários padrões de hiperparâmetros.
    - Caso o tipo não seja identificado explicitamente, retorna um modelo Keras Sequential vazio e os hiperparâmetros padrão para BiLSTM.
    - Pode ser facilmente expandida para novos tipos de modelos, bastando adicionar novos casos ao bloco condicional.

    Exemplo
    -------
    >>> best_model, best_params = get_best_model(some_fitted_estimator)
    >>> print(best_model)
    >>> print(best_params)
    """
    
    if isinstance(best, type(knn_model())):
        model = knn_model()
        parameters = std_knn_param()
    
    elif isinstance(best, type(lda_model())):
        model = lda_model()
        parameters = std_lda_param()
    
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


def optmize_model (X, y, model, no_tune=False, n_iter=1, refits=['roc_auc'], n_splits=10, n_repeats=1, early_stopping_rounds=15, random_state=6):

    """
    Realiza ajuste de hiperparâmetros e treinamento do modelo utilizando validação cruzada repetida e busca aleatória (RandomizedSearchCV).

    Esta função automatiza o processo de seleção de hiperparâmetros para diferentes tipos de modelos (incluindo redes neurais),
    podendo realizar desde o simples ajuste direto (sem tuning) até uma busca exploratória de parâmetros, com suporte para early stopping
    em modelos XGBoost e redes Keras. Retorna o melhor modelo ajustado conforme a métrica especificada.

    Parâmetros
    ----------
    X : pandas.DataFrame ou array-like
        Matriz de dados de entrada (features).
    y : array-like
        Vetor alvo (classes).
    model : estimador
        Instância do modelo base a ser otimizado. Pode ser XGBoost, KNN, LDA, SVC ou Sequential (Keras).
    no_tune : bool, opcional (default=False)
        Se True, o ajuste será feito diretamente, sem busca de hiperparâmetros (utiliza defaults).
    n_iter : int, opcional (default=1)
        Número de iterações na busca aleatória de hiperparâmetros.
    refits : list of str, opcional (default=['roc_auc'])
        Lista de métricas para refit após busca; o melhor modelo será escolhido conforme cada métrica.
    n_splits : int, opcional (default=10)
        Número de folds na validação cruzada.
    n_repeats : int, opcional (default=1)
        Número de repetições da validação cruzada (RepeatedKFold).
    early_stopping_rounds : int, opcional (default=15)
        Número de rounds para early stopping em modelos que suportam esse parâmetro (ex: XGBoost, BiLSTM).
    random_state : int, opcional (default=6)
        Semente para reprodução dos resultados.

    Retorno
    -------
    classifier_result : sklearn.model_selection.RandomizedSearchCV
        Instância do RandomizedSearchCV ajustada ao melhor conjunto de hiperparâmetros segundo a métrica de refit escolhida.

    Observações
    -----------
    - Suporta ajuste automático de parâmetros para múltiplos modelos; inclui pré-processamento para Keras (reshape do X e clear_session).
    - O pipeline é compatível com métricas customizadas (ex: specificity, F1 modificado).
    - No caso de `no_tune=True`, os hiperparâmetros defaults do modelo são usados.
    - O parâmetro `refits` pode ser uma lista de métricas; por padrão, usa 'roc_auc'.
    - Para redes Keras, utiliza EarlyStopping durante o treinamento para evitar overfitting.

    Exemplo
    -------
    >>> clf_result = optmize_model(X, y, model, n_iter=20, refits=['roc_auc', 'accuracy'])
    >>> print(clf_result.best_params_)
    """

    X = X.values
    y = y.astype(np.float64)

    scoring = scoring_params()

    k_fold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state) 

    best_model, parameters = get_best_model(model)

    if no_tune:
        parameters = {}

    if isinstance(best_model, type(xgb_model)):
        best_model.fit = partial(
            best_model.fit,
            early_stopping_rounds=early_stopping_rounds, #modificar para capturar do input
            eval_set=[(X, y)],
            verbose=0
        )

    if isinstance(best_model, Sequential):
        tf.keras.backend.clear_session()
        timesteps = 1
        X = X.reshape((X.shape[0], timesteps, X.shape[1]))

        model = build_bilstm_model(dropout_rate=0.3, dropout_dense=0.2, learning_rate=0.001, lstm_units=64, activation='tanh', input_shape=(timesteps, X.shape[2]))

        # Early stopping
        es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=5, restore_best_weights=True)

        best_model = KerasClassifier(
            model=build_bilstm_model,
            model__input_shape=(timesteps, X.shape[2]),
            epochs=10000,
            batch_size=64,
            verbose=0,
            callbacks=[es],
        )

    for refit in refits:
        
        classifier = RandomizedSearchCV(
            estimator=best_model,
            param_distributions=parameters,
            n_iter=n_iter,  
            scoring=scoring,
            n_jobs=-1,
            cv=k_fold,
            refit=refit,
            return_train_score=True,
            random_state=random_state
        )

        classifier_result = classifier.fit(X, y)
    
    return classifier_result


