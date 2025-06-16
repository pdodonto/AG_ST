
import numpy as np
import pandas as pd

import sys
sys.path.append("..")

import utils.utils as utils 
import utils.AG_FTIR as ag

import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import mahalanobis

def detect_outliers_mahalanobis(X, n_components=2, threshold=3):
    """
    Detecta outliers em um conjunto de dados utilizando a distância de Mahalanobis no espaço reduzido por PCA.

    O procedimento consiste em projetar os dados nas primeiras componentes principais (via PCA), calcular a distância
    de Mahalanobis de cada amostra ao centro da distribuição, e classificar como outlier os pontos cuja distância ultrapasse
    o limiar definido pelo parâmetro `threshold`.

    Parâmetros
    ----------
    X : array-like ou pd.DataFrame
        Dados de entrada (amostras x variáveis). Pode ser DataFrame ou ndarray.
    n_components : int, opcional (default=2)
        Número de componentes principais a serem mantidos na projeção PCA antes do cálculo da distância.
    threshold : float, opcional (default=3)
        Valor de corte para a distância de Mahalanobis. Amostras com distância maior ou igual ao limiar são consideradas outliers.

    Retorno
    -------
    X[mask] : array-like ou pd.DataFrame
        Subconjunto dos dados de entrada considerados como não-outliers (distância < threshold).
    mask : np.ndarray (boolean)
        Máscara booleana indicando quais amostras são consideradas válidas (não-outliers).

    Observações
    -----------
    - A função projeta os dados via PCA antes de calcular as distâncias de Mahalanobis, o que pode aumentar robustez em dados de alta dimensionalidade.
    - Útil para filtragem de outliers em análises espectrais (como FTIR), dados biomédicos e aplicações de pré-processamento para ML.
    - O limiar padrão (threshold=3) é equivalente ao usual para detecção de outliers (similar ao desvio padrão em distribuições normais).

    Exemplo
    -------
    >>> X_clean, mask = detect_outliers_mahalanobis(X, n_components=3, threshold=2.5)
    >>> print(X_clean.shape)
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    mean = X_pca.mean(axis=0)
    cov = np.cov(X_pca.T)
    cov_inv = np.linalg.inv(cov)
    
    distances = np.array([
        mahalanobis(x, mean, cov_inv) for x in X_pca
    ])
    mask = distances < threshold
    return X[mask], mask


def detect_outliers_isolation_forest(X, contamination=0.01):
    """
    Detecta outliers em um conjunto de dados utilizando o algoritmo Isolation Forest.

    O Isolation Forest é um método de aprendizado não supervisionado eficiente para identificação de anomalias,
    especialmente em grandes volumes de dados multidimensionais. Ele "isola" outliers mais rapidamente, pois são menos frequentes e diferentes dos demais pontos.

    Parâmetros
    ----------
    X : array-like ou pd.DataFrame
        Dados de entrada (amostras x variáveis). Pode ser DataFrame ou ndarray.
    contamination : float, opcional (default=0.01)
        Proporção estimada de outliers no conjunto de dados. Define a fração de amostras que serão consideradas anomalias.

    Retorno
    -------
    X[mask] : array-like ou pd.DataFrame
        Subconjunto dos dados de entrada considerados como não-outliers (inliers).
    mask : np.ndarray (boolean)
        Máscara booleana indicando quais amostras são inliers (True para não-outlier).

    Observações
    -----------
    - O método utiliza random_state=42 para reprodutibilidade.
    - 'inliers' são as amostras rotuladas com 1, enquanto outliers recebem -1 pelo algoritmo.
    - Útil para filtragem de outliers em pré-processamento de dados, análises espectrais (como FTIR) e bases biomédicas.

    Exemplo
    -------
    >>> X_clean, mask = detect_outliers_isolation_forest(X, contamination=0.02)
    >>> print(X_clean.shape)
    """
    iso = IsolationForest(contamination=contamination, random_state=42)
    y_pred = iso.fit_predict(X)
    mask = y_pred == 1  # 1 = inlier, -1 = outlier
    return X[mask], mask


# def apply_preprocessing(X,y,region=[[(1800,900)]]): #region=[[(1800,900)],[(3050,2800),(1800,900)]]
def apply_preprocessing(X,y,region=[[(1800,900)],[(3050,2800),(1800,900)]]):
    """
    Aplica etapas de pré-processamento espectral em dados FTIR, incluindo seleção de regiões espectrais, filtro de Savitzky-Golay
    e normalização pela banda Amida I, retornando diferentes versões dos dados para uso em pipelines de machine learning.

    Parâmetros
    ----------
    X : pd.DataFrame
        Dados espectrais (amostras x números de onda/colunas).
    y : array-like
        Vetor de classes ou variáveis alvo, associado às amostras de X.
    region : list, opcional
        Lista de listas de tuplas, cada tupla define um intervalo espectral (em cm⁻¹) a ser selecionado dos dados.
        Exemplo: [[(1800,900)], [(3050,2800),(1800,900)]]

    Retorno
    -------
    normalized : list of pd.DataFrame
        Versões dos dados normalizados pela banda Amida I (quando possível).
    savgol : list of pd.DataFrame
        Versões suavizadas com filtro de Savitzky-Golay (derivada 0).
    savgol_df : list of pd.DataFrame
        Versões suavizadas com Savitzky-Golay (primeira derivada).
    savgol_df_2 : list of pd.DataFrame
        Versões suavizadas com Savitzky-Golay (segunda derivada).

    Observações
    -----------
    - Para cada região definida em `region`, a função recorta os intervalos do DataFrame `X` e aplica o pré-processamento em sequência.
    - As funções auxiliares `utils.get_index`, `utils.apply_savgol` e `utils.normalize_amida_I` devem estar implementadas.
    - A normalização Amida I só é aplicada se a região (1700-1600 cm⁻¹) estiver presente.
    - A baseline correction está marcada para implementação futura.
    - Retorna múltiplas versões dos dados, prontos para experimentação comparativa em modelos ML/DL.

    Exemplo
    -------
    >>> norm, sg, sg1, sg2 = apply_preprocessing(X, y, region=[[(3050,2800),(1800,900)]])
    >>> print(norm[0].shape, sg[0].shape, sg1[0].shape, sg2[0].shape)
    """
    
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
    """
    Aplica a Análise de Componentes Principais (PCA) ao conjunto de treino e projeta o conjunto de teste no mesmo espaço de componentes principais.

    Parâmetros
    ----------
    FTIR_data : pd.DataFrame ou array-like
        Dados de treino (amostras x variáveis), geralmente espectros FTIR.
    test_data : pd.DataFrame ou array-like
        Dados de teste a serem projetados no mesmo espaço PCA.
    num_components : int ou float
        Número de componentes principais a serem mantidos (ou fração de variância explicada, se float entre 0 e 1).

    Retorno
    -------
    x_t : pd.DataFrame
        Dados de treino transformados no espaço dos componentes principais.
    t_t : np.ndarray
        Dados de teste projetados no mesmo espaço PCA.
    pca : sklearn.decomposition.PCA
        Objeto PCA ajustado no conjunto de treino.

    Exemplo
    -------
    >>> X_train_pca, X_test_pca, pca = pca_transform(X_train, X_test, num_components=0.95)
    >>> print(X_train_pca.shape, X_test_pca.shape)
    """
    pca = PCA(n_components=num_components)
    x_t = pd.DataFrame(pca.fit_transform(FTIR_data))
    t_t = pca.transform(test_data)

    return x_t, t_t, pca


def pca_multi_transform(X_bases, X_test_bases, vars=[0.95, 0.99, 0.999, 0.9999]):
    """
    Aplica PCA em múltiplas bases de dados (ou regiões espectrais), para diferentes níveis de variância explicada,
    transformando tanto o conjunto de treino quanto o de teste para cada caso.

    Parâmetros
    ----------
    X_bases : list of pd.DataFrame
        Lista de bases de treino (cada base pode representar uma região espectral diferente).
    X_test_bases : list of pd.DataFrame
        Lista correspondente de bases de teste.
    vars : list of float, opcional
        Lista de valores para a variância explicada acumulada (entre 0 e 1), usados para selecionar o número de componentes do PCA.

    Retorno
    -------
    pca_data : list
        Lista de listas, onde cada elemento corresponde a uma base e contém tuplas com:
            (X_t, T_t, descrição, objeto_pca, colunas_originais)
        - X_t: treino transformado,
        - T_t: teste transformado,
        - descrição: string resumindo a base e a variância explicada,
        - objeto_pca: instância PCA ajustada,
        - colunas_originais: nomes das colunas da base de teste.

    Observações
    -----------
    - Permite aplicar pipelines de PCA sobre diferentes regiões espectrais ou abordagens de segmentação.
    - Facilita comparação do efeito de diferentes níveis de redução de dimensionalidade.

    Exemplo
    -------
    >>> pca_data = pca_multi_transform([X1, X2], [X1_test, X2_test], vars=[0.95, 0.99])
    >>> X_train_pca, X_test_pca, desc, pca_obj, cols = pca_data[0][0]
    """

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


@st.cache_data(show_spinner=False)
def compute_mnf(X_train, X_test):
    """
    Realiza a transformação MNF (Minimum Noise Fraction) em múltiplas bases de dados, 
    armazenando o resultado em cache para acelerar execuções repetidas no Streamlit.

    Parâmetros
    ----------
    X_train : list of pd.DataFrame
        Lista de bases de treino (cada base pode ser uma região espectral diferente).
    X_test : list of pd.DataFrame
        Lista correspondente de bases de teste.

    Retorno
    -------
    mnf_data : list
        Lista de listas, onde cada elemento contém tuplas com os dados MNF transformados para cada nível de variância.
        Ver documentação da função `mnf_multi_transform`.

    Observações
    -----------
    - Utiliza cache do Streamlit para evitar reprocessamento quando os dados de entrada não mudam.
    - Encapsula chamada à função `mnf_multi_transform`.

    Exemplo
    -------
    >>> mnf_data = compute_mnf([X1, X2], [X1_test, X2_test])
    """
    return mnf_multi_transform(X_train, X_test)

def mnf_multi_transform(X_bases, X_test_bases, vars=[0.95, 0.99, 0.999, 0.9999]):
    """
    Aplica a transformação MNF em múltiplas bases/regiões espectrais, para diferentes níveis de variância explicada,
    tanto para os dados de treino quanto para os dados de teste, retornando estruturas organizadas para experimentação e análise.

    Parâmetros
    ----------
    X_bases : list of pd.DataFrame
        Lista de bases de treino (amostras x variáveis), normalmente correspondendo a diferentes regiões espectrais ou abordagens.
    X_test_bases : list of pd.DataFrame
        Lista correspondente de bases de teste.
    vars : list of float, opcional
        Lista de valores para a variância acumulada desejada em cada transformação MNF.

    Retorno
    -------
    mnf_data : list
        Lista de listas, onde cada elemento corresponde a uma base e contém tuplas com:
            (X_t, T_t, descrição, objeto_pca, colunas_originais)
        - X_t: treino transformado,
        - T_t: teste transformado,
        - descrição: string com informação da base e variância,
        - objeto_pca: instância PCA/MNF ajustada,
        - colunas_originais: nomes das colunas da base de teste.

    Observações
    -----------
    - Utiliza barra de progresso do Streamlit para acompanhamento visual.
    - Em cada base, para cada valor de variância, seleciona o número ideal de componentes via função auxiliar,
      aplica a transformação MNF e armazena todos os resultados organizados.
    - Facilita a análise comparativa do efeito da MNF em diferentes regiões e com diferentes níveis de redução de dimensionalidade.

    Exemplo
    -------
    >>> mnf_data = mnf_multi_transform([X1, X2], [X1_test, X2_test], vars=[0.95, 0.99])
    >>> X_train_mnf, X_test_mnf, desc, pca_obj, cols = mnf_data[0][0]
    """
    mnf_data = []
    for i in range(len(X_bases)):
        mnf_bar = st.progress(0)
        mnf_var = []
        count = 1
        for var in vars:
           num_components = utils.select_mnf_plot(X_bases[i],  cum_var=var, show_cum_var=False, plotar=False, return_n_comp=True)
           x_t, t_t, pca = utils.mnf_new_prop(FTIR_data=X_bases[i], validation=X_test_bases[i], num_components=num_components)
           mnf_bar.progress(count/len(vars), text = f'var: {var} - n_comp: {num_components}')
           mnf_var.append((pd.DataFrame(x_t), pd.DataFrame(t_t), f'MNF({i}): {var} - init_cols({X_bases[i].shape[1]})', pca, X_test_bases[i].columns))
           count += 1
        mnf_data.append(mnf_var)
        mnf_bar.empty()
    return mnf_data


@st.cache_data(show_spinner=False)
def compute_smnf(X_train, X_test, segments=None):
    """
    Executa a transformação SMNF (Segmented/Strategic Minimum Noise Fraction) em múltiplas bases de treino e teste,
    utilizando cache do Streamlit para acelerar execuções repetidas.

    Parâmetros
    ----------
    X_train : list of pd.DataFrame
        Lista de bases de treino (cada base pode corresponder a diferentes regiões espectrais).
    X_test : list of pd.DataFrame
        Lista correspondente de bases de teste.
    segments : array-like ou None, opcional
        Segmentos espectrais pré-definidos ou None para seleção automática/estratégica.

    Retorno
    -------
    smnf_data : list
        Lista estruturada com os dados SMNF transformados para cada base e cada nível de variância explicada,
        conforme produzido pela função `smnf_multi_transform`.

    Observações
    -----------
    - Utiliza cache de dados do Streamlit para maior desempenho em execuções repetidas.
    - Ideal para aplicações com múltiplas execuções em interfaces interativas.

    Exemplo
    -------
    >>> smnf_data = compute_smnf([X1, X2], [X1_test, X2_test], segments=segments)
    """
    return smnf_multi_transform(X_train, X_test, segments=segments)


def smnf_multi_transform(X_bases, X_test_bases, segments, vars=[0.999]):
    """
    Aplica a transformação SMNF (Segmented/Strategic Minimum Noise Fraction) em múltiplas bases de treino e teste,
    para diferentes níveis de variância explicada, utilizando segmentação espectral fornecida.

    Parâmetros
    ----------
    X_bases : list of pd.DataFrame
        Lista de bases de treino (amostras x variáveis), normalmente de diferentes regiões espectrais.
    X_test_bases : list of pd.DataFrame
        Lista correspondente de bases de teste.
    segments : array-like
        Segmentos espectrais definidos manualmente ou por estratégias automáticas para aplicação do SMNF.
    vars : list of float, opcional
        Lista de valores para a variância acumulada desejada em cada transformação SMNF.

    Retorno
    -------
    smnf_data : list
        Lista de listas, onde cada elemento corresponde a uma base e contém tuplas com:
            (X_t, T_t, descrição, objeto_pca, colunas_originais)
        - X_t: treino transformado,
        - T_t: teste transformado,
        - descrição: string com informação da base e variância,
        - objeto_pca: instância PCA/MNF ajustada,
        - colunas_originais: nomes das colunas da base de teste.

    Observações
    -----------
    - Exibe barra de progresso no Streamlit durante o processamento de cada base e cada nível de variância.
    - Utiliza função auxiliar `utils.mnf_strategic_new_prop` para aplicar a transformação SMNF baseada nos segmentos definidos.
    - Útil para análise e validação comparativa de estratégias de redução de dimensionalidade segmentada em dados espectrais, especialmente FTIR.

    Exemplo
    -------
    >>> smnf_data = smnf_multi_transform([X1, X2], [X1_test, X2_test], segments=segments, vars=[0.99, 0.999])
    >>> X_train_smnf, X_test_smnf, desc, pca_obj, cols = smnf_data[0][0]
    """
    smnf_data = []
    for i in range(len(X_bases)):
        smnf_bar = st.progress(0)
        smnf_var = []
        count = 1
        for var in vars:
           num_components = utils.select_mnf_plot(X_bases[i],  cum_var=var, show_cum_var=False, plotar=False, return_n_comp=True)
           x_t, t_t, pca = utils.mnf_strategic_new_prop(FTIR_data=X_bases[i], test_data=X_test_bases[i], segments=segments, cum_var=var)
           smnf_bar.progress(count/len(vars), text = f'var: {var} - n_comp: {num_components}')
           smnf_var.append((pd.DataFrame(x_t), pd.DataFrame(t_t), f'SMNF({i}): {var} - init_cols({X_bases[i].shape[1]})', pca, X_test_bases[i].columns))
           count += 1
        smnf_data.append(smnf_var)
        smnf_bar.empty()
    return smnf_data


def conjugate_bases(a, b, value):
    """
    Constrói uma lista de tuplas combinando pares de DataFrames correspondentes de duas listas (a e b),
    associando cada par a uma descrição textual com a faixa espectral.

    Parâmetros
    ----------
    a : list of pd.DataFrame
        Primeira lista de DataFrames, tipicamente bases de treino.
    b : list of pd.DataFrame
        Segunda lista de DataFrames, tipicamente bases de teste (ou validação) correspondentes.
    value : str
        Texto ou etiqueta identificadora do tipo/base (ex: 'PCA', 'MNF', etc.), usada na descrição.

    Retorno
    -------
    base : list of tuples
        Lista de tuplas com (DataFrame de a, DataFrame de b, descrição textual da faixa espectral).
        Exemplo: (a[i], b[i], 'PCA: (wavenumber_start - wavenumber_end)')

    Exemplo
    -------
    >>> base = conjugate_bases(list_a, list_b, 'PCA')
    >>> print(base[0][2])
    'PCA: (4000.0 - 900.0)'
    """
    base = []
    for i in range(len(a)):
        base.append((a[i], b[i], f'{value}: ({a[i].columns[0]} - {a[i].columns[-1]})'))
    return base

def pca_choices(model, bases, y):
    """
    Seleciona, entre diferentes pares de bases (ex: conjuntos PCA para diferentes regiões),
    aquela com maior fitness, de acordo com a avaliação de um modelo e função de fitness (por exemplo, acurácia).

    Parâmetros
    ----------
    model : estimador
        Modelo de machine learning a ser avaliado (ex: KNN, LDA, XGB, etc.).
    bases : list of tuples
        Lista de bases, onde cada elemento é uma tupla (DataFrame de treino, DataFrame de teste, descrição).
        Tipicamente gerada por `conjugate_bases`.
    y : array-like
        Vetor alvo (classes) correspondente ao conjunto de treino.

    Retorno
    -------
    pca_choosen : tuple
        A tupla (DataFrame treino, DataFrame teste, descrição) da base com maior fitness avaliado.
        Se várias bases tiverem o mesmo fitness máximo, retorna a primeira encontrada.

    Observações
    -----------
    - Utiliza função de fitness do módulo 'ag', baseada no modelo fornecido e seleção aleatória de features.
    - Útil para automatizar a seleção da melhor região/componentes/função de redução de dimensionalidade para um determinado problema.

    Exemplo
    -------
    >>> best_base = pca_choices(model, bases, y)
    >>> print(best_base[2])
    'PCA: (1800.0 - 900.0)'
    """
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
    return pca_choosen



