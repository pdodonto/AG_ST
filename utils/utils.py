import numpy as np
from sklearn.decomposition import PCA

import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

from scipy import signal
from sklearn.covariance import LedoitWolf

from scipy import sparse
from scipy.sparse.linalg import spsolve

#testar
# from pybaselines import als
from scipy.signal import find_peaks

# Metric
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score



def orange_mnf (X, num_components):
    '''
    Implementação MNF Orange
    '''
    m = X.shape

    diffs = -np.diff(X, axis=0)
    N = diffs

    # noise whitening
    V1, S, _ = np.linalg.svd(np.dot(N.T, N))
    # s1 = s1 ** -0.5
    S = np.linalg.inv(np.sqrt(np.diag(S)))
    W = np.dot(np.dot(X, V1), S)

    # directions maximizing signal variance
    G1, _, _ = np.linalg.svd(np.dot(W.T, W))

    # Final MNF projection vectors calculation
    P = np.dot(np.dot(V1, S), G1)

    # X data on the MNF directions space
    M = np.dot(X, P)

    # choice of top components
    R = np.eye(m[1], m[1])
    R[num_components:, num_components:] = 0

    # inverse MNF transformation
    D = np.dot(np.dot(M, R), np.linalg.inv(P))

    return D

def aplicar_mnf_validacao(X_val, P, num_components=None):
    """
    Aplica a projeção MNF nos dados de validação usando a matriz P treinada.
    
    Parâmetros:
        X_val: array (n_amostras_validação, n_wavenumbers)
        P: matriz de projeção MNF obtida dos dados de treino
        num_components: número de componentes MNF desejadas (mesmo do treino)

    Retorna:
        M_val: dados de validação no espaço MNF
        D_val: dados reconstruídos (denoised)
    """
    M_val = X_val @ P  # Projeta no espaço MNF
    
    # Redução de dimensionalidade (se necessário)
    if num_components is not None:
        M_val = M_val[:, :num_components]
    
        # Para reconstrução: preenche com zeros as componentes descartadas
        M_val_full = np.zeros((X_val.shape[0], P.shape[1]))
        M_val_full[:, :num_components] = M_val
    else:
        M_val_full = M_val
    
    # Reconstrução no espaço original
    P_inv = np.linalg.pinv(P)
    D_val = M_val_full @ P_inv

    return M_val, D_val


def selecionar_componentes_mnf(M, variancia_min=0.95, plotar=True, ax=None):
    """
    Seleciona o número ideal de componentes MNF com base na variância acumulada e cotovelo.
    
    Parâmetros:
        M: Dados projetados no espaço MNF (n_amostras, n_componentes)
        variancia_min: Variância acumulada mínima desejada (ex: 0.95 = 95%)
        plotar: Se True, plota o gráfico da variância explicada
        ax: (opcional) Eixo do matplotlib para plotar (útil para subplots)

    Retorna:
        n_variancia: número de componentes com base no limiar de variância
    """
    # Calcula a variância por componente
    variancia = np.var(M, axis=0)
    variancia_normalizada = variancia / np.sum(variancia)
    variancia_acumulada = np.cumsum(variancia_normalizada)

    # Critério 1: Variância acumulada mínima
    n_variancia = np.searchsorted(variancia_acumulada, variancia_min) + 1

    # Gráfico (opcional)
    if plotar:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        
        ax.plot(variancia_acumulada, label='Variância Acumulada')
        ax.axvline(n_variancia, color='g', linestyle='--', label=f'{variancia_min*100:.3f}% da variância: {n_variancia}')

        ax.set_xlabel('Número de Componentes MNF')
        ax.set_ylabel('Variância Acumulada')
        ax.legend()
        ax.grid(True)

        if ax is None:
            plt.tight_layout()
            plt.show()

    return n_variancia


def reconstruct_mnf(M, P, num_components):
    """
    Reconstrói os dados originais a partir da projeção MNF.

    Parâmetros:
        M: dados no espaço MNF (n_amostras, n_variáveis)
        P: matriz de projeção MNF (n_variáveis, n_variáveis)
        num_components: número de componentes MNF a manter

    Retorna:
        D: dados reconstruídos no espaço original (denoised)
    """
    m = P.shape[0]
    R = np.eye(m)
    if num_components < m:
        R[num_components:, num_components:] = 0

    D = M @ R @ np.linalg.inv(P)
    return D


def estimate_noise(data, method='first_diff', window_size=3):
    """
    Estima o ruído com base no método especificado.
    """
    if method == 'first_diff':
        return -np.diff(data, axis=0)
    
    elif method == 'moving_avg':
        def moving_average(x):
            return np.convolve(x, np.ones(window_size)/window_size, mode='same')
        smooth = np.apply_along_axis(moving_average, axis=1, arr=data)
        return data - smooth

    elif method == 'pca_residual':
        pca = PCA(n_components=0.99)
        X_recon = pca.inverse_transform(pca.fit_transform(data))
        return data - X_recon

    else:
        raise ValueError("Método de ruído inválido. Use 'first_diff', 'moving_avg' ou 'pca_residual'.")

def mnf_transform_modular(FTIR_data, num_components=None, noise_method='first_diff', window_size=3):
    """
    MNF transform com método de ruído ajustável (Orange, com base em Bhargava, 2018).
    
    Parâmetros:
        FTIR_data: array (n_amostras, n_wavenumbers)
        num_components: nº de componentes MNF desejadas
        noise_method: método para estimativa do ruído
        window_size: parâmetro para média móvel (se aplicável)

    Retorna:
        D: dados reconstruídos (denoised)
        M: dados no espaço MNF
        P: matriz de projeção MNF
    """

    # Etapa 1: Estima o ruído
    N = estimate_noise(FTIR_data, method=noise_method, window_size=window_size)

    # Etapa 2: Whitening via SVD
    V1, S_noise, _ = np.linalg.svd(N.T @ N)
    S_inv_sqrt = np.linalg.inv(np.sqrt(np.diag(S_noise)))
    W = FTIR_data @ V1 @ S_inv_sqrt

    # Etapa 3: Direções que maximizam variação do sinal
    G1, _, _ = np.linalg.svd(W.T @ W)

    # Etapa 4: Matriz de projeção MNF
    P = V1 @ S_inv_sqrt @ G1

    # Etapa 5: Projeção MNF
    M = FTIR_data @ P

    # Etapa 6: Seleção de componentes
    m = FTIR_data.shape[1]
    R = np.eye(m)
    if num_components is not None and num_components < m:
        R[num_components:, num_components:] = 0

    # Etapa 7: Reconstrução
    D = M @ R @ np.linalg.inv(P)

    return D, M, P, W


#==================================================================================================
# Função para dividir em blocos
def split_into_blocks(array, block_size):
    n_blocks = array.shape[1] // block_size
    return np.array_split(array, n_blocks, axis=1)

def estimate_block_based_covariance(data, block_size:int=50):
    # Escolher tamanho do bloco (ex: 200 wavenumbers por bloco)
    block_size = block_size
    blocks = split_into_blocks(data, block_size) 

    # Estimar covariância por bloco usando Ledoit-Wolf para estabilidade
    cov_matrices = []
    for block in blocks:
        lw = LedoitWolf()
        lw.fit(block)
        cov_matrices.append(lw.covariance_)
    
    return cov_matrices


def estimate_noise_covariance_first_diff(data):
    # Estima o ruído pela diferença entre bandas consecutivas
    diff = -np.diff(data, axis=0)
    noise_cov = np.cov(diff, rowvar=False)
    epsilon = 1e-10 #pequeno valor adicionado para evitar problemas numéricos (testar novamente)
    noise_cov += np.eye(noise_cov.shape[0]) * epsilon 

    return noise_cov
    

def estimate_noise_covariance_moving_avg(data, window_size=3):
    # Estima o ruído subtraindo a média móvel por amostra
    def moving_average(x):
        return np.convolve(x, np.ones(window_size)/window_size, mode='same')
    # smooth = np.apply_along_axis(np.convolve(data, np.ones(window_size)/window_size, mode='same'), axis=1, arr=data)
    smooth = np.apply_along_axis(moving_average, axis=1, arr=data)
    residual = data - smooth
    noise_cov = np.cov(residual, rowvar=False)
    epsilon = 1e-10 #epsilon = 1e-10
    noise_cov += np.eye(noise_cov.shape[0]) * epsilon  

    return noise_cov

def estimate_noise_covariance_pca_residual(data, n_signal_components=5):
    # Estima o ruído assumindo que as primeiras PCs representam o sinal
    pca = PCA(n_components=n_signal_components)
    signal = pca.inverse_transform(pca.fit_transform(data))
    residual = data - signal
    noise_cov = np.cov(residual, rowvar=False)
    epsilon = 1e-10
    noise_cov += np.eye(noise_cov.shape[0]) * epsilon  

    return noise_cov

def mnf_option_transform(ftir_data, num_components=None, random_state=None, noise_method='first_diff', block_size:int=1):
    """
    Aplica a transformação MNF nos dados de FTIR.

    Parâmetros:
        ftir_data: numpy array de forma (amostras, bandas) representando espectros FTIR.
        num_components: número de componentes MNF a serem mantidos. Se None, mantém todos.
        random_state: valor de random_state para reprodução.
        noise_method: método de estimativa de ruído ('first_diff', 'moving_avg', 'pca_residual').
        block_size: valor do tamanho de bloco para estimar a matriz de covariância do ruído no método block_based.

    Retorna:
        mnf_components: espectros transformados MNF.
        pca: modelo PCA treinado.
        whitening_transform: matriz de branqueamento.
        reconstruct_mnf: espectros reconstruídos, com base no número de componentes selecionados.
    """
    # Passo 1: Estima a matriz de covariância do ruído
    if noise_method == 'first_diff':
        noise_cov = estimate_noise_covariance_first_diff(ftir_data)
    elif noise_method == 'moving_avg':
        noise_cov = estimate_noise_covariance_moving_avg(ftir_data)
    elif noise_method == 'pca_residual':
        noise_cov = estimate_noise_covariance_pca_residual(ftir_data)
    elif noise_method == 'block_based':
        noise_cov = estimate_block_based_covariance(ftir_data, block_size)
    else:
        raise ValueError("Método de ruído inválido. Use 'first_diff', 'moving_avg', 'pca_residual' ou 'block_based(teste).")

    # Passo 2: Branqueamento dos dados para remover correlação do ruído
    eigvals, eigvecs = np.linalg.eigh(noise_cov)
    whitening_transform = np.linalg.inv(eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T)
    whitened_data = ftir_data @ whitening_transform.T  # Remove correlação do ruído

    # Passo 3: Aplica PCA nos dados transformados, caso definido
    if num_components:
        pca = PCA(n_components=num_components, random_state=random_state)
        mnf_components = pca.fit_transform(whitened_data)
        reconstruct_mnf = pca.inverse_transform(mnf_components) @ np.linalg.inv(whitening_transform)
    else:
        pca = None
        mnf_components = whitened_data
        reconstruct_mnf = ftir_data

    return mnf_components, pca, whitening_transform, reconstruct_mnf


def mnf_transform_apply(ftir_data, pca, whitening_transform):
    """
    Aplica a transformacão MNF em dados novos (validação ou teste).
    Requer o PCA e o branqueamento treinados no conjunto de treino.
    """
    whitened_data = ftir_data @ whitening_transform.T
    if pca:
        mnf_transformed = pca.transform(whitened_data)
    else:
        mnf_transformed = whitened_data
    return mnf_transformed


#================================================================================================================


def open_file (path:str, columns:np.array=None, invert_columns:bool=False, get_classes:bool=False):
    """ 
    Abre um arquivo e o transforma em um DataFrame Pandas; separa a última coluna para classes.

    Parâmetros:
        path: string com o caminho.
        columns: colunas a serem consideradas.
        invert_columns: Se deseja inverter o espectro.
        get_classes: Caso verdadeiro, separa a última coluna para classes.
    
    Retorna:
        data: Dados em DataFrame Pandas
        classes: 
    """
    file = open(path, 'r')
    data = pd.read_table(file, header=None)

    if get_classes:
        classes = np.array(data.iloc[:,-1]).flatten()
        data = data.iloc[:,:-1]
    else:
        classes = None

    if columns.any():
        data.columns = columns
    
    if invert_columns:
        data = data.iloc[:,::-1]
    
    return data,classes if get_classes  else data


def get_index (data:pd.DataFrame, wave:float):
    """
    Retorna o índice do número de onda, dado um DataFrame 
    """
    try:
        wave = [x.astype(np.str_) for x in data.columns.values.astype(np.float64) if x >= wave and x < wave+1]
        wave = data.columns.get_indexer(wave)[0]
        return wave
    except:
        raise ValueError('número de onda não encontrado')


def get_class_index (data:np.array):
    """
    Verifica o número de classes no array e retorna os índices
    """
    info = {}
    classes = np.unique(data)
    for i in range(0,classes.size):
        info.update({classes[i]:np.where(data == classes[i])[0]})
    
    return info


def plot_spectrum(data:list, color:list, label:list=['Dados'], legend_fontsize:int=8, xticks:list=[], first_columns:bool=True, last_columns:bool=True, 
                  rotation:int=0, fontsize:int=8, figsize:tuple=(10,6), alpha:float=0.25, linewidth:int=1, ylabel:str='Absorbância', 
                  xlabel:str='Número de onda'+r'($cm^{-1}$)', column_values:bool=True):
    """
    Plota um gráfico com base nos parâmetros passados

    Parâmetros:
        data: bases de dados no formato lista.
        label: lista de labels.
        color: lista de cores para cada base de dados na lista.
        xticks: lista com sticks desejados.
        rotation: rotação de eixo dos xticks.
        figsize: tupla indicando o tamanho da figura.
        alpha: valor float de transparência.
        ylabel: string com o nome do eixo y.
        xlabel: string com o nome do eixo x.
        
    """
    plt.figure(figsize=figsize)
    for i in range(0,len(data)):
        for index in range(0, len(data[i])):

            if index != len(data[i]) - 1:
                plt.plot((data[i].columns).astype('str'), np.array(data[i].iloc[index]), color = color[i], alpha = alpha, linewidth=linewidth)
            else:
                plt.plot((data[i].columns).astype('str'), np.array(data[i].iloc[index]), label = label[i], color = color[i], alpha = alpha, linewidth=linewidth)
        
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)

    for i in range(len(data)):
        try:
            if first_columns:
                xticks = np.append(xticks, [(data[i].columns[0]).astype('str')])
            if last_columns:
                xticks = np.append(xticks, [(data[i].columns[-1]).astype('str')])
        except:
            if first_columns:
                xticks = np.append(xticks, [data[i].columns[0]])
            if last_columns:
                xticks = np.append(xticks, [data[i].columns[-1]])
            
    
    plt.xticks(fontsize=fontsize)
    plt.xticks(rotation=rotation)
    plt.xticks(xticks)
    
    plt.legend(fontsize=legend_fontsize)

def plot_spectrum_streamlit(data:list, color:list, label:list=['Dados'], legend_fontsize:int=8, xticks:list=[], first_columns:bool=True, last_columns:bool=True, 
                  rotation:int=0, fontsize:int=8, figsize:tuple=(10,6), alpha:float=0.25, linewidth:int=1, ylabel:str='Absorbância', 
                  xlabel:str='Número de onda'+r'($cm^{-1}$)', column_values:bool=True):
    """
    Plota um gráfico com base nos parâmetros passados e retorna o objeto `fig`.

    Parâmetros:
        data: lista de DataFrames contendo os espectros.
        color: lista de cores para cada base de dados.
        label: lista de labels para a legenda.
        xticks: lista de posições personalizadas no eixo x.
        rotation: rotação dos valores no eixo x.
        figsize: tamanho da figura.
        alpha: transparência das linhas.
        linewidth: espessura das linhas.
        ylabel: rótulo do eixo y.
        xlabel: rótulo do eixo x.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for i in range(len(data)):
        for index in range(len(data[i])):
            col_str = data[i].columns.astype(str)
            y_vals = np.array(data[i].iloc[index])
            if index != len(data[i]) - 1:
                ax.plot(col_str, y_vals, color=color[i], alpha=alpha, linewidth=linewidth)
            else:
                ax.plot(col_str, y_vals, label=label[i], color=color[i], alpha=alpha, linewidth=linewidth)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    xticks_out = list(xticks) 
    for i in range(len(data)):
        try:
            if first_columns:
                xticks_out.append(str(data[i].columns[0]))
            if last_columns:
                xticks_out.append(str(data[i].columns[-1]))
        except:
            pass

    ax.set_xticks(xticks_out)
    ax.tick_params(axis='x', rotation=rotation, labelsize=fontsize)
    ax.legend(fontsize=legend_fontsize)

    return fig


# Utilização visual para confirmação da suspeita de outlier
def visual_pca_plot(X):
    if isinstance(X, pd.DataFrame):
        X = np.array(X.values)

    X_flat = X.reshape(X.shape[0], -1)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)

    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    plt.title("PCA - visualização em projeção 2D (por variação total)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()

def visual_pca_plot_streamlit(X):
    if isinstance(X, pd.DataFrame):
        X = X.values  # converte para ndarray se for DataFrame

    X_flat = X.reshape(X.shape[0], -1)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)

    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

    return df_pca


def peak_normalization (data:pd.DataFrame, first_index:int, second_index:int ):
    """
    Normaliza a base de dados com base no pico do intervalo espectral passado como parâmetro.

    Parâmetros:
        data: base de dados FTIR
        first_index: índice do primeiro intervalo
        second_index: índie do segundo intervalo
    
    Retorna:
        normalized_df: Base normalizada com base no intervalo considerado
    """
    # Extrai a região espectral e pega o valor absoluto (módulo)
    peak_region = data.iloc[:, first_index:second_index].abs()
    # Calcula o valor máximo do pico (valor absoluto) para cada amostra
    max_peak = peak_region.max(axis=1)
    # Evitar divisão por zero
    max_peak[max_peak == 0] = 1e-10
    # Normaliza dividindo cada linha pelo valor máximo do pico (abs)
    normalized_df = data.div(max_peak, axis=0)

    return normalized_df


# Correção de Baseline
def baseline_als_batch(X, lam=1e6, p=0.001, niter=10):
    """
    Aplica ALS para correção de baseline linha a linha em um conjunto de espectros FTIR.
    
    Parâmetros:
        X     : array (n_amostras, n_wavenumbers)
        lam   : parâmetro de suavização (lambda)
        p     : parâmetro de assimetria (penalização)
        niter : número de iterações

    Retorna:
        X_corrigido : espectros com baseline removido
        baselines   : baseline estimado para cada espectro
    """
    n_amostras, n_wavenumbers = X.shape
    baselines = np.zeros_like(X)
    X_corrigido = np.zeros_like(X)

    # Monta a matriz de penalização (diferenças de 2ª ordem)
    L = n_wavenumbers
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L))
    D = D.dot(D.transpose())

    for i in range(n_amostras):
        y = X[i]
        w = np.ones(L)
        for _ in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        baselines[i] = z
        X_corrigido[i] = y - z

    return X_corrigido, baselines


def select_mnf_plot(total:pd.DataFrame, num_components:int=0, noise_method:str='moving_avg', random_state:int=42, cum_var:float=0.95, show_cum_var:bool=True, plotar:bool=True, ax=None, return_n_comp:bool=False):
    """
    Aplica a transformação MNF (Minimum Noise Fraction) a um conjunto de dados e auxilia na seleção do número
    de componentes com base na variância acumulada. Também gera gráficos para análise visual.

    Parâmetros:
    ------------
    total : pd.DataFrame
        Conjunto de dados (por exemplo, espectros FTIR) a ser transformado pela MNF.
    
    num_components : int, opcional (default=0)
        Número de componentes principais desejado para a transformação.
        Se 0, usa o mínimo entre o número de amostras e o número de variáveis.
    
    noise_method : str, opcional (default='moving_avg')
        Método utilizado para estimar o ruído no processo de MNF. Exemplos: 'moving_avg', 'gaussian', etc.
    
    random_state : int, opcional (default=42)
        Semente de aleatoriedade para garantir reprodutibilidade do processo de MNF.
    
    cum_var : float, opcional (default=0.95)
        Variância acumulada desejada (entre 0 e 1) para seleção automática do número de componentes mais relevantes.
    
    show_cum_var : bool, opcional (default=True)
        Se True, imprime no terminal o número de componentes necessários para atingir o valor de variância acumulada desejado.
    
    plotar : bool, opcional (default=True)
        Se True, gera um gráfico da variância explicada para ajudar na seleção dos componentes.

    Retorna:
    --------
    None
        A função é utilizada para visualização e apoio à decisão sobre o número ideal de componentes MNF.
    
    Notas:
    ------
    - Utiliza internamente `utils.mnf_option_transform` para realizar a transformação MNF.
    - Utiliza `utils.selecionar_componentes_mnf` para selecionar componentes e gerar gráficos.
    - A impressão da variância acumulada é útil para selecionar de forma objetiva o número de componentes a manter.

    """
    if num_components == 0 or num_components > min(total.shape):
        num_components = min(total.shape)
    mnf_result_a2, pca_a2, whitening_transform,reconstruct_mnf = mnf_option_transform(total.values, num_components=num_components, noise_method=noise_method, random_state=random_state)

    
    value = 0
    n_comp = 0
    for i in pca_a2.explained_variance_ratio_:
        if value <= cum_var:
            n_comp += 1
            value += i
            if show_cum_var:
                print(f'n_comp:{n_comp}, value:{value}')
        else:
            break
  
    selecionar_componentes_mnf(mnf_result_a2,cum_var,plotar=plotar,ax=ax)

    if return_n_comp == True:
        return n_comp


def mnf_data_transform(base:pd.DataFrame, val_base:pd.DataFrame, num_components:int=0, noise_method:str='moving_avg', random_state:int=42):
    """
    Aplica a transformação MNF (Minimum Noise Fraction) aos dados de treino e validação.

    A transformação MNF é utilizada para reduzir ruído e dimensionalidade em conjuntos de dados espectroscópicos,
    como dados de FTIR ou hiperespectrais. Este método aplica uma transformação PCA seguida de um
    branqueamento para maximizar a separação entre sinal e ruído.

    Parâmetros:
    ------------
    base : pd.DataFrame
        Conjunto de dados principal (treinamento) a ser transformado.
    
    val_base : pd.DataFrame
        Conjunto de dados de validação externa a ser transformado com os mesmos parâmetros da base.
    
    num_components : int, opcional (default=0)
        Número de componentes principais a serem mantidos na transformação.
        Se 0, usa o mínimo entre o número de amostras e de variáveis como número de componentes.

    noise_method : str, opcional (default='moving_avg')
        Método utilizado para estimar o ruído nos dados. Exemplos comuns: 'moving_avg', 'gaussian', etc.
    
    random_state : int, opcional (default=42)
        Semente para reprodutibilidade da decomposição MNF.

    Retorna:
    --------
    mnf_result_a2 : np.ndarray
        Dados de treinamento transformados por MNF.
    
    mnf_result_validation : np.ndarray
        Dados de validação transformados pela mesma transformação MNF aplicada à base.
    """
    if num_components == 0:
        num_components = min(base.shape)

    mnf_result_a2, pca_a2, whitening_transform,reconstruct_mnf = mnf_option_transform(base.values, num_components=num_components, noise_method=noise_method, random_state=random_state)
    mnf_result_validation = mnf_transform_apply(val_base.values, pca_a2, whitening_transform)

    return mnf_result_a2, mnf_result_validation


def mnf_stratsel(base, selection: int = 0, v_distance: int = 30, v_height: float = -0.99, 
                 v_prominence: float = 0.0007, p_distance: int = 5, p_height: float = 0.02, 
                 p_prominence: float = 0.02, plot: int = 3):
    """
    Realiza a seleção de regiões espectrais (estratificação) em dados FTIR 
    com base na detecção de picos e vales.

    Parâmetros:
    ----------
    base : pd.DataFrame
        Base de dados FTIR (linhas: amostras, colunas: número de onda).
    selection : int, opcional (default=0)
        Define o tipo de marcação:
            0 - Usar apenas vales,
            1 - Usar apenas picos,
            2 - Usar combinação de picos e vales.
    v_distance : int, opcional (default=30)
        Distância mínima entre vales detectados (em número de pontos).
    v_height : float, opcional (default=-0.99)
        Altura mínima para considerar um vale (valores negativos para inverter o espectro).
    v_prominence : float, opcional (default=0.0007)
        Proeminência mínima para considerar um vale.
    p_distance : int, opcional (default=5)
        Distância mínima entre picos detectados (em número de pontos).
    p_height : float, opcional (default=0.02)
        Altura mínima para considerar um pico.
    p_prominence : float, opcional (default=0.02)
        Proeminência mínima para considerar um pico.
    plot : int, opcional (default=3)
        Define o tipo de gráfico gerado:
            0 - Apenas vales,
            1 - Apenas picos,
            2 - Ambos (picos e vales),
            3 - Todos os gráficos.

    Retorna:
    -------
    mark : np.ndarray
        Vetor de índices indicando os pontos de divisão das regiões espectrais
        (útil para segmentação posterior dos dados).

    Notas:
    -----
    - A função utiliza `find_peaks` da `scipy.signal` tanto para picos como para vales.
    - Inverte o eixo X no gráfico, conforme a convenção de espectros FTIR.
    - A seleção pode ser feita apenas por picos, vales ou ambos.
    - Utilizado para pré-processar dados antes do método MNF.
    """
    y = -base.mean().values
    x = base.columns.astype(np.float64)
    # Detecção de vales
    valleys, v_properties = find_peaks(y, height=v_height, prominence=v_prominence, distance=v_distance)

    y = base.mean().values

    if plot == 0 or plot == 3:
        # Plot com picos destacados
        plt.figure(figsize=(16, 8))
        plt.plot(x, y, label="Espectro FTIR", color='blue')
        plt.plot(x[valleys], y[valleys], 'go', label='Vales Detectados')
        plt.xlabel("Número de onda (cm$^{-1}$)")
        plt.ylabel("Absorbância")
        plt.legend()
        plt.grid(True)
        plt.gca().invert_xaxis()  # espectros FTIR normalmente têm eixo invertido
        plt.show()

    y = base.mean().values
    x = base.columns.astype(np.float64)
    # Detecção de picos
    peaks, p_properties = find_peaks(y, height=p_height, prominence=p_prominence, distance=p_distance)

    if plot == 1 or plot == 3:
        # Plot com picos destacados
        plt.figure(figsize=(16, 8))
        plt.plot(x, y, label="Espectro FTIR", color='blue')
        plt.plot(x[peaks], y[peaks], 'ro', label='Picos Detectados')
        plt.xlabel("Número de onda (cm$^{-1}$)")
        plt.ylabel("Absorbância")
        plt.legend()
        plt.grid(True)
        plt.gca().invert_xaxis() 
        plt.show()
    
    if plot == 2 or plot == 3:
        # Plot com picos destacados
        plt.figure(figsize=(16, 8))
        plt.plot(x, y, label="Espectro FTIR", color='blue')
        plt.plot(x[valleys], y[valleys], 'go', label='vales Detectados')
        plt.plot(x[peaks], y[peaks], 'ro', label='Picos Detectados')
        plt.xlabel("Número de onda (cm$^{-1}$)")
        plt.ylabel("Absorbância")
        plt.legend()
        plt.grid(True)
        plt.gca().invert_xaxis()  # espectros FTIR normalmente têm eixo invertido
        plt.show()

    if selection == 0:
        segments = valleys
    elif selection == 1:
        segments = peaks
    else:
        segments = np.sort(np.concatenate((peaks, valleys), axis=0))

    mark = np.add(1,segments)
    segments = np.append([0], segments)
    segments = np.append(segments, len(base.columns))
    mark = np.append([0], mark)
    mark = np.append(mark, len(base.columns))
        
    return mark


def mnf_region_segmentation(base:pd.DataFrame, segments:np.ndarray, print_sum:bool=True):
    """
    Segmenta a base de dados FTIR de acordo com os índices fornecidos.

    Cada segmento é criado com base nos índices de início e fim passados em `segments`,
    gerando subconjuntos da base para análise local em regiões específicas do espectro.

    Parâmetros:
        base (pd.DataFrame): Base de dados FTIR, onde linhas são amostras e colunas são números de onda.
        segments (np.ndarray ou list): Vetor de índices indicando o início e fim de cada segmento.

    Retorna:
        specs (list of pd.DataFrame): Lista de DataFrames, cada um representando uma região segmentada do espectro.

    Observação:
        - Um print progressivo mostra o número acumulado de bandas consideradas até cada segmento.
        - Ao final, imprime o número total de segmentos criados.
    """
    specs = []
    sum = 0

    for i in range(segments.size - 1):
        r = pd.DataFrame(base.values[:, segments[i]:segments[i+1]])
        r.columns = base.columns[segments[i]:segments[i+1]]
        sum += len(r.columns)
        if print_sum:
            print(sum, len(r.columns))
        specs.append(r)

    if print_sum:    
        print(f"Número de segmentos: {len(specs)}")

    return specs


def get_spec_pca_whitening(specs, components=None, use_components=0):
    """
    Aplica o processo de MNF (Maximum Noise Fraction) em cada segmento espectral fornecido.

    Para cada segmento, realiza:
      - Transformação PCA (Principal Component Analysis).
      - Transformação de branqueamento (whitening).
      - Cálculo da reconstrução MNF.

    Parâmetros:
        specs (list of pd.DataFrame): Lista de segmentos espectrais (DataFrames) a serem processados.

    Retorna:
        specs_mnf (list of tuples): Lista contendo tuplas (mnf_result, reconstruct_mnf) para cada segmento:
            - mnf_result: Dados transformados no espaço MNF.
            - reconstruct_mnf: Dados reconstruídos a partir da transformação MNF.
        pca_mnf (list): Lista com os objetos PCA ajustados para cada segmento.
        whitening_transform_mnf (list): Lista com as matrizes de transformação de branqueamento de cada segmento.

    Observações:
        - A função utiliza o método `mnf_option_transform` para aplicar a sequência PCA + Whitening + MNF.
        - O número de componentes usado é o mesmo número de bandas do respectivo segmento.
        - Usa como método de estimativa de ruído o 'moving_avg'.
        - Define o `random_state=42` para garantir reprodutibilidade.
    """
    specs_mnf = []
    pca_mnf = []
    whitening_transform_mnf = []

    for i in range(len(specs)):
        if use_components == 1:
            num_components = components[i].shape[1]
        else:
            num_components = specs[i].shape[1]
        mnf_result, pca, whitening_transform, reconstruct_mnf = mnf_option_transform(
            specs[i].values, 
            num_components=num_components, 
            noise_method='moving_avg', 
            random_state=42
        )
        specs_mnf.append((mnf_result, reconstruct_mnf))
        pca_mnf.append(pca)
        whitening_transform_mnf.append(whitening_transform)
    
    return specs_mnf, pca_mnf, whitening_transform_mnf


def get_spec_pca_transformed(pca_mnf, specs_mnf, value=0.99, n_components=None):
    """
    Seleciona o número de componentes MNF necessários para atingir uma variância acumulada desejada 
    em cada segmento espectral, e concatena os componentes selecionados em um único DataFrame.

    Parâmetros:
        pca_mnf (list): Lista dos objetos PCA ajustados para cada segmento (não utilizado diretamente na função, mas mantido para consistência de assinatura).
        specs_mnf (list of tuples): Lista contendo tuplas (mnf_result, reconstruct_mnf) para cada segmento:
            - mnf_result: Dados transformados no espaço MNF.
        value (float, opcional): Valor de variância acumulada mínima desejada para seleção de componentes.
            Default é 0.99 (99% da variância explicada).

    Retorna:
        n_comp (list of int): Lista contendo o número de componentes selecionados para cada segmento.
        transformed (pd.DataFrame): DataFrame concatenado contendo os componentes MNF selecionados de todos os segmentos.
            As colunas são renumeradas de 0 até o total de componentes concatenados.

    Observações:
        - A função utiliza `utils.selecionar_componentes_mnf` para determinar o número de componentes com base na variância acumulada.
        - Cada segmento é reduzido até o número de componentes correspondente e então todos os segmentos são unidos lado a lado.
        - A variável `pca_mnf` está presente na assinatura mas atualmente não é utilizada dentro da função.
    """
    n_comp = []
    transformed = []
    var = []
    for i in range(len(pca_mnf)):
        if n_components == None:
            var = selecionar_componentes_mnf(specs_mnf[i][0], value, plotar=False)
            n_comp.append(var)
        else:
            var = n_components[i]
            n_comp.append(var)
        df = pd.DataFrame(specs_mnf[i][0])
        transformed.append(df.iloc[:, 0:var])

    transformed = pd.concat(transformed, axis=1)
    transformed.columns = np.arange(np.sum(n_comp))

    return n_comp, transformed


def apply_new_mnf_transform(base, segments, n_comp, pca_mnf, whitening_transform_mnf):
    """
    Aplica a transformação MNF (Minimum Noise Fraction) em segmentos específicos da base de dados,
    utilizando PCA e transformações de whitening previamente treinados.

    Parâmetros:
    ----------
    base : pd.DataFrame
        Base de dados original (ex.: espectros FTIR) com as amostras nas linhas e características nas colunas.
    
    segments : np.ndarray ou list
        Vetor de índices que define as regiões (segmentos) de divisão da base. 
        Ex: [0, 30, 60, 100] separa em três regiões: 0-29, 30-59, 60-99.
    
    n_comp : list
        Lista contendo o número de componentes a serem utilizados em cada segmento após o MNF.
    
    pca_mnf : list
        Lista dos modelos PCA ajustados (um para cada segmento).
    
    whitening_transform_mnf : list
        Lista das matrizes de whitening associadas a cada modelo PCA (um para cada segmento).

    Retorna:
    -------
    transformed : pd.DataFrame
        Base transformada, contendo apenas os componentes selecionados de todos os segmentos, concatenados.
        As colunas são renumeradas de 0 até (soma total dos componentes - 1).
    
    Observações:
    ------------
    - A função primeiro divide a base nos segmentos fornecidos.
    - Depois aplica a transformação MNF em cada segmento usando os modelos PCA e whitening treinados.
    - Apenas os componentes desejados (definidos em `n_comp`) de cada segmento são concatenados no resultado final.
    """
    
    specs = mnf_region_segmentation(base, segments)

    transformed = []
    for i in range(len(specs)):
        r = pd.DataFrame(mnf_transform_apply(specs[i], pca_mnf[i], whitening_transform_mnf[i]))
        transformed.append(r.iloc[:, 0:n_comp[i]])
    
    transformed = pd.concat(transformed, axis=1)
    transformed.columns = np.arange(np.sum(n_comp))
    
    return transformed








#=====================================================================================#
##Pipeline de pre-processamento##
def correct_baseline(spectra, lam=1e5, p=0.01, niter=10):
    """
    Corrige o baseline dos espectros utilizando o método ALS (Asymmetric Least Squares).

    Parâmetros:
    ----------
    spectra : pd.DataFrame
        DataFrame contendo os espectros (linhas = amostras, colunas = bandas).
    lam : float
        Parâmetro de suavização (lambda) para o ALS.
    p : float
        Parâmetro de assimetria para o ALS (valores próximos de 0 enfatizam picos positivos).
    niter : int
        Número de iterações do algoritmo ALS.

    Retorna:
    -------
    pd.DataFrame
        Espectros com baseline corrigido.
    """
    baseline = []
    corrected = []
    for i in range(spectra.shape[0]):
        base, _ = als.als(spectra.iloc[i, :], lam=lam, p=p, niter=niter)
        baseline.append(base)
        corrected.append(spectra.iloc[i, :] - base)
    corrected_df = pd.DataFrame(corrected, columns=spectra.columns, index=spectra.index)
    return corrected_df

def apply_savgol(spectra, window_length=11, polyorder=2, deriv=1):
    """
    Aplica filtro de Savitzky-Golay para suavização ou derivação dos espectros.

    Parâmetros:
    ----------
    spectra : pd.DataFrame
        DataFrame com espectros (linhas = amostras, colunas = bandas).
    window_length : int
        Comprimento da janela do filtro (precisa ser ímpar).
    polyorder : int
        Ordem do polinômio para ajuste dentro da janela.
    deriv : int
        Ordem da derivada a ser aplicada (0 para suavização).

    Retorna:
    -------
    pd.DataFrame
        Espectros processados com Savitzky-Golay.
    """
    deriv_spectra = signal.savgol_filter(spectra, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1)
    deriv_df = pd.DataFrame(deriv_spectra, columns=spectra.columns, index=spectra.index)
    return deriv_df

def normalize_amida_I(spectra, wavenumbers):
    """
    Normaliza os espectros dividindo pela intensidade máxima da região da Amida I (1600–1700 cm⁻¹).

    Parâmetros:
    ----------
    spectra : pd.DataFrame
        DataFrame com espectros (linhas = amostras, colunas = bandas).
    wavenumbers : np.ndarray
        Array de números de onda correspondentes às colunas do espectro.

    Retorna:
    -------
    pd.DataFrame
        Espectros normalizados pela intensidade máxima na região da Amida I.
    """
    amida_region = (wavenumbers >= 1600) & (wavenumbers <= 1700)
    normalized = []
    for i in range(spectra.shape[0]):
        amida_max = np.max(np.abs(spectra.iloc[i, amida_region]))
        normalized.append(spectra.iloc[i, :] / amida_max)
    normalized_df = pd.DataFrame(normalized, columns=spectra.columns, index=spectra.index)
    return normalized_df

def preprocess_ftir(spectra, wavenumbers, truncate_min=900, truncate_max=1800,
                    baseline_params={'lam': 1e5, 'p': 0.01, 'niter': 10},
                    savgol_params={'window_length': 11, 'polyorder': 2, 'deriv': 1}):
    """
    Pipeline completo para pré-processamento de dados FTIR:
    - Truncamento da região espectral
    - Correção de baseline (ALS)
    - Derivada (Savitzky-Golay)
    - Normalização pela Amida I

    Parâmetros:
    ----------
    spectra : pd.DataFrame
        DataFrame com espectros (linhas = amostras, colunas = bandas).
    wavenumbers : np.ndarray
        Array de números de onda correspondentes às colunas do espectro.
    truncate_min : float
        Valor mínimo de número de onda para truncamento.
    truncate_max : float
        Valor máximo de número de onda para truncamento.
    baseline_params : dict
        Parâmetros para correção de baseline.
    savgol_params : dict
        Parâmetros para filtro de Savitzky-Golay.

    Retorna:
    -------
    pd.DataFrame
        Espectros pré-processados.
    """
    # 1. Truncamento da região espectral
    mask = (wavenumbers >= truncate_min) & (wavenumbers <= truncate_max)
    spectra_truncated = spectra.loc[:, mask]
    
    # 2. Correção de baseline
    # spectra_corrected = correct_baseline(spectra_truncated, **baseline_params)
    spectra_corrected = spectra_truncated

    # 3. Aplicação do filtro de Savitzky-Golay
    spectra_deriv = apply_savgol(spectra_corrected, **savgol_params)
    
    # 4. Normalização pela Amida I
    spectra_normalized = normalize_amida_I(spectra_deriv, wavenumbers[mask])
    
    return spectra_normalized



#-------------------nova proposição--------------------------------------------------

def pca_new_prop(FTIR_data, test_data, num_components, plot=0):
    FTIR_data = FTIR_data
    test_data = test_data
    pca = PCA(n_components=num_components)
    transformed = pd.DataFrame(pca.fit_transform(FTIR_data))

    x = np.array([])
    for i in range(test_data.shape[0]):
        data = pd.concat([FTIR_data,pd.DataFrame(test_data.iloc[i]).T], ignore_index=True)
        
        data_transformed = pd.DataFrame(pca.fit_transform(data))

        data = data_transformed
        if i == 0:
            x = data.iloc[-1].values
        else:
            x = np.vstack((x,data.iloc[-1].values))

    test_data = pd.DataFrame(x)
    
    if plot == 0:
        plot_spectrum([transformed],['blue'],['pca'])
    transformed.shape

    return transformed, test_data, pca

def mnf_new_prop(FTIR_data, validation, num_components, method='moving_avg'):
    FTIR_data = FTIR_data
    validation = validation
    num_components = num_components
    method = 'moving_avg'
    transformed, pca, whitening_transform,_ = mnf_option_transform(FTIR_data, num_components=num_components, noise_method=method, random_state=42)

    x = np.array([])
    for i in range(validation.shape[0]):
        data = pd.concat([FTIR_data,pd.DataFrame(validation.iloc[i]).T], ignore_index=True)
        
        data_transformed, pca_a2, whitening_transform,_ = mnf_option_transform(data, num_components=num_components, noise_method=method, random_state=42)

        data = pd.DataFrame(data_transformed)
        if i == 0:
            x = data.iloc[-1].values
        else:
            x = np.vstack((x,data.iloc[-1].values))

    val_data = pd.DataFrame(x)
    transformed = pd.DataFrame(transformed)
    
    return transformed, val_data, pca

def mnf_strategic_new_prop(FTIR_data, validation, cum_var, v_distance=100, plot=0):
    FTIR_data = FTIR_data
    validation = validation
    cum_var = cum_var

    segments = mnf_stratsel(FTIR_data,selection=0, plot=0, v_distance=v_distance) #Escolha dos vales (selection=0) e sem plot(plot=-1)
    specs = mnf_region_segmentation(FTIR_data, segments)
    specs_mnf, pca_mnf, whitening_transform_mnf = get_spec_pca_whitening(specs)
    n_comp, transformed = get_spec_pca_transformed(pca_mnf, specs_mnf, cum_var)

    x = np.array([])
    for i in range(validation.shape[0]):
        data = pd.concat([FTIR_data,pd.DataFrame(validation.iloc[i]).T], ignore_index=True)

        specs_data = mnf_region_segmentation(data, segments, print_sum=False)
        specs_mnf, pca_mnf, whitening_transform_mnf = get_spec_pca_whitening(specs_data)
        _, data_transformed = get_spec_pca_transformed(pca_mnf, specs_mnf, cum_var, n_components=n_comp)

        if i == 0:
            x = data_transformed.iloc[-1].values
        else:
            x = np.vstack((x,data_transformed.iloc[-1].values))

    val_data = pd.DataFrame(x)
    print(transformed.shape)

    return transformed, val_data