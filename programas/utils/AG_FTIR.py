import numpy as np
import tensorflow as tf
import pandas as pd
import streamlit as st
from sklearn.model_selection import cross_val_score
import copy
import utils.models as models
from scikeras.wrappers import KerasClassifier
from keras.callbacks import EarlyStopping


def gen_initial_pop(size:int, Tp:int=10, p_1:float=0.5, random_prob:bool=False):
    """
    Gera uma população inicial de indivíduos para algoritmos genéticos ou métodos evolutivos.

    Cada indivíduo é representado por um vetor binário, onde cada gene é sorteado conforme uma probabilidade especificada.

    Parâmetros
    ----------
    size : int
        Tamanho do vetor (quantidade de genes) de cada indivíduo.
    Tp : int, opcional (default=10)
        Número de indivíduos (população total) a serem gerados.
    p_1 : float, opcional (default=0.5)
        Probabilidade de um gene assumir o valor 1 (o complemento será a probabilidade de ser 0).
    random_prob : bool, opcional (default=False)
        Se True, utiliza uma probabilidade diferente e aleatória de 'p_1' para cada indivíduo.

    Retorna
    -------
    pop : list of numpy.ndarray
        Lista contendo os indivíduos gerados, cada um representado por um vetor binário (numpy array).
    
    Exemplo
    -------
    >>> pop = gen_initial_pop(size=5, Tp=3, p_1=0.2)
    >>> for ind in pop:
    ...     print(ind)
    [0 0 0 0 1]
    [0 1 0 0 0]
    [0 0 0 0 0]
    """
    pop = []

    for i in range(Tp):
        if random_prob:
            p_1 = np.random.random_sample()
        p_0 = 1 - p_1
        individual = np.random.choice([0, 1], size=size, p=[p_0, p_1])
        pop.append(individual)

    return pop



def fitness_func(model, solution:np.array, X:pd.DataFrame, y:np.array, min_features:int=5, cv:int=5, scoring:str='accuracy'):
    """
    Calcula a aptidão (fitness) de uma solução de seleção de atributos em um pipeline de modelagem supervisionada.

    Essa função avalia o desempenho de um modelo ao treinar e validar apenas um subconjunto de atributos selecionados
    a partir da máscara binária da solução.

    Parâmetros
    ----------
    model : estimator ou objeto com atributo 'best_estimator_'
        O modelo de machine learning a ser avaliado. Pode ser um estimador treinado ou um resultado de GridSearchCV, RandomizedSearchCV, etc.
        Suporte especial para KerasClassifier (SciKeras ou Keras Wrappers).
    solution : np.array
        Máscara binária indicando os atributos selecionados (1 para selecionado, 0 para descartado).
    X : pd.DataFrame
        Conjunto de dados (features) original.
    y : np.array
        Vetor de classes/variáveis alvo.
    min_features : int, opcional (default=5)
        Número mínimo de atributos selecionados exigido para a avaliação da solução.
        Caso a solução possua menos atributos que esse limiar, retorna 0 como fitness.
    cv : int, opcional (default=5)
        Número de folds (partições) na validação cruzada.
    scoring : str, opcional (default='accuracy')
        Métrica de avaliação usada na validação cruzada (por exemplo: 'accuracy', 'roc_auc', 'f1', etc.).

    Retorno
    -------
    score : float
        Score médio obtido na validação cruzada para a solução proposta. Retorna 0 se o número de atributos for menor que `min_features`.

    Observações
    -----------
    - Para modelos do tipo KerasClassifier (rede neural), são realizados procedimentos adicionais de reshape dos dados
      e reinicialização dos parâmetros antes da avaliação, visando garantir compatibilidade com a arquitetura sequencial.
    - Caso o modelo seja proveniente de busca em grid (possua 'best_estimator_'), a avaliação é realizada sobre o melhor estimador encontrado.
    - Retorna a média dos scores nas folds da validação cruzada.

    Exemplo
    -------
    >>> score = fitness_func(model, solution, X, y, min_features=5, cv=5, scoring='accuracy')
    >>> print(f"Fitness: {score:.3f}")
    """
    mask = np.array(solution, dtype=bool)
    if np.sum(mask) < min_features:  #seleção mínima de features
        return 0
    X_selected = X.iloc[:, mask]
    
    try:
        if isinstance(model.best_estimator_, KerasClassifier):
            tf.keras.backend.clear_session()
            timesteps = 1
            X_selected = X_selected.values.reshape((X_selected.values.shape[0], timesteps, X_selected.values.shape[1]))
            params = model.best_params_
            es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=5, restore_best_weights=True)

            model = KerasClassifier(
                model=models.build_bilstm_model,
                **params,
                model__input_shape=(timesteps, X_selected.shape[2]),
                epochs=10000,
                verbose=0,
                callbacks=[es],
            )
    except:
        pass

    try:
        scores = cross_val_score(model.best_estimator_, X_selected, y, cv=cv, scoring=scoring)
    except:
        scores = cross_val_score(model, X_selected, y, cv=cv, scoring=scoring)

    return scores.mean()



def get_fitness(model, pop, FTIR_data:pd.DataFrame, y:np.array, cv:int=5):
    """
    Avalia a aptidão (fitness) de uma população de soluções de seleção de atributos, utilizando validação cruzada.

    Esta função percorre todos os indivíduos da população, calcula o fitness (desempenho) de cada solução (máscara binária)
    com base em um modelo predefinido e retorna uma lista de tuplas (indivíduo, fitness). Exibe uma barra de progresso
    interativa utilizando Streamlit para acompanhamento em tempo real do processo.

    Parâmetros
    ----------
    model : estimador ou objeto com atributo 'best_estimator_'
        O modelo de machine learning a ser avaliado em cada subconjunto de atributos.
    pop : list of np.array
        População de soluções, onde cada indivíduo é uma máscara binária indicando as features selecionadas.
    FTIR_data : pd.DataFrame
        Conjunto de dados de entrada (features), tipicamente espectros FTIR.
    y : np.array
        Vetor de classes/variável alvo.
    cv : int, opcional (default=5)
        Número de folds na validação cruzada.

    Retorno
    -------
    fitness : list of tuples
        Lista contendo tuplas do tipo (indivíduo, fitness), em que o fitness é o score médio obtido na validação cruzada
        para aquela máscara de seleção de atributos.

    Observações
    -----------
    - Utiliza barra de progresso Streamlit para feedback visual do processo.
    - O cálculo do fitness de cada indivíduo é feito através da função `fitness_func`, que avalia o modelo somente nas features selecionadas.
    - Recomendado para uso em algoritmos evolutivos/genéticos de seleção de atributos.

    Exemplo
    -------
    >>> fitness_list = get_fitness(model, pop, FTIR_data, y, cv=5)
    >>> for ind, fit in fitness_list:
    ...     print(f"Indivíduo: {ind}, Fitness: {fit:.3f}")
    """
    fitness = []
    my_bar = st.progress(0, text="Calculating fitness...")
    count = 0
    for individual in pop:
        fit = fitness_func(model, individual, FTIR_data, y, cv=cv)
        fitness.append((individual, fit))

        count += 1
        my_bar = my_bar.progress((count/len(pop)), text=f"Calculating fitness [{count}/{len(pop)}]...")
    my_bar.empty()
    return fitness


def order_by_fitness(pop, reverse:bool=True):
    """
    Ordena uma população de indivíduos de acordo com seus valores de fitness.

    Esta função recebe uma lista de tuplas (indivíduo, fitness) e retorna a lista ordenada pelo valor de fitness,
    de forma decrescente (por padrão), ou crescente, conforme o parâmetro.

    Parâmetros
    ----------
    pop : list of tuples
        Lista onde cada elemento é uma tupla contendo um indivíduo e seu respectivo valor de fitness: (indivíduo, fitness).
    reverse : bool, opcional (default=True)
        Se True, ordena do maior para o menor fitness (decrescente). Se False, do menor para o maior (crescente).

    Retorno
    -------
    fit_sorted : list of tuples
        Lista ordenada de tuplas (indivíduo, fitness), conforme especificado.

    Exemplo
    -------
    >>> pop = [([0, 1, 1], 0.85), ([1, 0, 0], 0.65), ([1, 1, 1], 0.92)]
    >>> ordered = order_by_fitness(pop)
    >>> for ind, fit in ordered:
    ...     print(f"Fitness: {fit:.2f} - Indivíduo: {ind}")
    Fitness: 0.92 - Indivíduo: [1, 1, 1]
    Fitness: 0.85 - Indivíduo: [0, 1, 1]
    Fitness: 0.65 - Indivíduo: [1, 0, 0]
    """
    fit_sorted = sorted(pop, key=lambda x: x[1], reverse=reverse)
    return fit_sorted



def truncation_selection(pop, p_cross:float=0.4):
    """
    Realiza a seleção por truncamento para cruzamento no algoritmo genético.

    A seleção por truncamento consiste em selecionar uma fração dos melhores indivíduos da população 
    (com base no valor de fitness) e, a partir desse grupo, formar pares aleatórios para cruzamento.

    Parâmetros
    ----------
    pop : list of tuples
        População de indivíduos, onde cada elemento é uma tupla (indivíduo, fitness).
    p_cross : float, opcional (default=0.4)
        Proporção dos melhores indivíduos a serem considerados para cruzamento, no intervalo (0, 1].

    Retorno
    -------
    pairs : list of tuples
        Lista de pares de indivíduos selecionados para cruzamento. Cada par é uma tupla de dois indivíduos (cada um no formato (indivíduo, fitness)).

    Exceções
    --------
    Exception
        Lança exceção caso p_cross não esteja no intervalo (0, 1].

    Observações
    -----------
    - Os pares são formados aleatoriamente entre os melhores indivíduos selecionados.
    - O número de pares é calculado como metade da quantidade de indivíduos selecionados (arredondado).

    Exemplo
    -------
    >>> pop = [([1,0,1], 0.95), ([1,1,1], 0.90), ([0,0,1], 0.70), ([0,1,0], 0.60)]
    >>> pairs = truncation_selection(pop, p_cross=0.5)
    >>> for p in pairs:
    ...     print(p)
    (([1, 0, 1], 0.95), ([1, 1, 1], 0.90))
    """
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
    """
    Realiza a seleção baseada em ranking para formação de pares.

    Nesta abordagem, os indivíduos são ordenados pelo valor de fitness, e a probabilidade de seleção é 
    proporcional ao ranking: indivíduos com melhor fitness têm maior chance de serem escolhidos para cruzamento.

    Parâmetros
    ----------
    pop : list of tuples
        População de indivíduos, onde cada elemento é uma tupla (indivíduo, fitness).
    pairs_number : int
        Número de pares a serem selecionados para cruzamento. Deve ser inteiro positivo.

    Retorno
    -------
    pairs : list of tuples
        Lista de pares de indivíduos selecionados para cruzamento. Cada par é uma tupla contendo dois indivíduos (cada um no formato (indivíduo, fitness)).

    Exceções
    --------
    Exception
        Lança exceção se `pairs_number` não for um inteiro positivo.

    Observações
    -----------
    - A probabilidade de seleção de cada indivíduo é calculada de forma linear de acordo com sua posição no ranking.
    - Os pares são formados sem reposição, ou seja, um mesmo indivíduo não pode compor o mesmo par.
    - O fitness dos indivíduos deve ser tal que quanto maior, melhor (por padrão, a lista é ordenada do maior para o menor fitness).

    Exemplo
    -------
    >>> pop = [([0,1,1], 0.80), ([1,0,0], 0.75), ([1,1,1], 0.90), ([0,0,1], 0.65)]
    >>> pairs = rank_based_selection(pop, pairs_number=2)
    >>> for p in pairs:
    ...     print(p)
    (([1, 1, 1], 0.90), ([0, 1, 1], 0.80))
    """
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

    """
    Realiza a seleção por roleta (roulette wheel) para formação de pares em algoritmos evolutivos/genéticos.

    Nesta abordagem, a chance de um indivíduo ser selecionado é proporcional ao seu valor de fitness, 
    simulando o giro de uma roleta onde indivíduos com maior fitness ocupam uma "fatia" maior da probabilidade.

    Parâmetros
    ----------
    pop : list of tuples
        População de indivíduos, onde cada elemento é uma tupla (indivíduo, fitness).
    pairs_number : int
        Número de pares a serem selecionados para cruzamento. Deve ser inteiro positivo.
    replace : bool, opcional (default=False)
        Se True, permite que o mesmo indivíduo seja selecionado mais de uma vez no mesmo par. Se False, pares sem repetição.

    Retorno
    -------
    pairs : list of tuples
        Lista de pares de indivíduos selecionados para cruzamento. Cada par é uma tupla contendo dois indivíduos (cada um no formato (indivíduo, fitness)).

    Exceções
    --------
    Exception
        Lança exceção se `pairs_number` não for positivo.
    ValueError
        Lança exceção se a soma dos valores de fitness for zero, pois a seleção por roleta não é possível nesse caso.

    Observações
    -----------
    - Indivíduos com fitness zero não são favorecidos, enquanto indivíduos com fitness mais alto têm maior probabilidade de seleção.
    - A seleção é feita proporcionalmente ao fitness relativo na população.

    Exemplo
    -------
    >>> pop = [([1,0,1], 0.95), ([1,1,1], 0.90), ([0,0,1], 0.70), ([0,1,0], 0.00)]
    >>> pairs = roulette_wheel_selection(pop, pairs_number=2)
    >>> for p in pairs:
    ...     print(p)
    (([1, 0, 1], 0.95), ([1, 1, 1], 0.90))
    """
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

    """
    Realiza o crossover (cruzamento) simples de um ponto entre pares de indivíduos.

    Para cada par de indivíduos, seleciona aleatoriamente um ponto de corte e gera dois novos indivíduos,
    combinando segmentos de genes dos "pais". Este operador é fundamental em algoritmos genéticos para
    criar diversidade na população.

    Parâmetros
    ----------
    pairs : list of tuples
        Lista de pares de indivíduos para cruzamento. Cada par é uma tupla contendo dois indivíduos,
        e cada indivíduo deve ser um array-like binário representando os genes.

    Retorno
    -------
    new_individuals : list of np.ndarray
        Lista com os novos indivíduos (filhos) gerados pelo crossover. Para cada par, são gerados dois filhos.

    Exceções
    --------
    ValueError
        Lança exceção se o número de genes em um indivíduo for menor que 2.

    Observações
    -----------
    - O ponto de corte é escolhido aleatoriamente entre 1 e (n_genes - 1), garantindo pelo menos um gene de cada "pai" nos filhos.
    - Não preserva o fitness dos pais ou dos filhos. Apenas retorna a nova população de indivíduos (representados apenas pelos genes).
    - O formato esperado para cada indivíduo é algo como: (array_genes, fitness), logo o acesso é feito via pair[0][0].

    Exemplo
    -------
    >>> import numpy as np
    >>> pai1 = (np.array([1, 0, 1, 1]), 0.92)
    >>> pai2 = (np.array([0, 1, 0, 0]), 0.89)
    >>> pairs = [(pai1, pai2)]
    >>> filhos = simple_crossover(pairs)
    >>> for filho in filhos:
    ...     print(filho)
    [1 0 0 0]
    [0 1 1 1]
    """
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
    """
    Realiza a mutação binária em uma população de indivíduos.

    Para cada gene de cada indivíduo, existe uma probabilidade `p_mut` de ocorrer mutação,
    ou seja, o valor do gene é invertido (de 0 para 1 ou de 1 para 0). 
    Utilizado em algoritmos genéticos para promover diversidade e evitar estagnação em ótimos locais.

    Parâmetros
    ----------
    pop : list of array-like
        População de indivíduos. Cada indivíduo deve ser um array binário representando os genes.
    p_mut : float, opcional (default=0.01)
        Probabilidade de mutação para cada gene. Valor entre 0 e 1.

    Retorno
    -------
    population : list of array-like
        Nova população resultante da aplicação da mutação. Os indivíduos originais não são alterados.

    Observações
    -----------
    - A função utiliza deepcopy para preservar os indivíduos originais.
    - A mutação ocorre de forma independente para cada gene.
    - É esperado que os indivíduos sejam arrays (np.ndarray, list, etc.) contendo apenas 0s e 1s.

    Exemplo
    -------
    >>> import numpy as np
    >>> pop = [np.array([1,0,1,1]), np.array([0,1,0,0])]
    >>> mutated_pop = binary_mutation(pop, p_mut=0.5)
    >>> for ind in mutated_pop:
    ...     print(ind)
    [1 1 0 1]
    [0 1 1 0]
    """
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
    """
    Realiza a reinserção ordenada (elitista) em algoritmos evolutivos/genéticos.

    Combina a população atual e a nova população gerada (por cruzamento, mutação, etc.), 
    ordena todos os indivíduos pelo valor de fitness e seleciona os `t_p` melhores para formar a nova geração.

    Parâmetros
    ----------
    pop : list of tuples
        População atual, onde cada elemento é uma tupla (indivíduo, fitness).
    new_pop : list of tuples
        Nova população de indivíduos, também como tuplas (indivíduo, fitness).
    t_p : int, opcional (default=0)
        Tamanho da nova geração. Se 0 ou inválido (menor que 1 ou maior que o tamanho total disponível), 
        assume o tamanho da população original (`len(pop)`).

    Retorno
    -------
    new_gen : list of tuples
        Nova geração, composta pelos `t_p` melhores indivíduos (de acordo com o fitness), 
        no formato de tuplas (indivíduo, fitness).

    Observações
    -----------
    - Mantém sempre os indivíduos de maior fitness, preservando a elitização.
    - Usa deepcopy para garantir que as populações de entrada não sejam alteradas.
    - Se o número de indivíduos desejado (`t_p`) for maior que o tamanho total, retorna o máximo possível.

    Exemplo
    -------
    >>> pop = [([1,0,1], 0.82), ([0,1,1], 0.75)]
    >>> new_pop = [([1,1,1], 0.91), ([0,0,1], 0.65)]
    >>> next_gen = sorted_reinsertion(pop, new_pop, t_p=3)
    >>> for ind, fit in next_gen:
    ...     print(f"Indivíduo: {ind}, Fitness: {fit:.2f}")
    Indivíduo: [1, 1, 1], Fitness: 0.91
    Indivíduo: [1, 0, 1], Fitness: 0.82
    Indivíduo: [0, 1, 1], Fitness: 0.75
    """
    full_pop = copy.deepcopy(pop) + copy.deepcopy(new_pop)
    if t_p <= 0 or t_p > len(full_pop):
        t_p = len(pop)
    pop_ordered = order_by_fitness(full_pop, reverse=True)

    new_gen = pop_ordered[:t_p]

    return new_gen



def ag_pipeline (X, y, model, generations:int=30, Tp:int=20, gen_random_prob:bool=True, pairs_number:int=3, cv:int=5, early_stopping:int=10, insert_self:bool=False):

    """
    Executa o pipeline de um Algoritmo Genético (AG) para seleção de atributos em aprendizado de máquina.

    Esta função aplica um ciclo completo de algoritmo genético — geração da população inicial, avaliação de fitness,
    seleção, crossover, mutação, reinserção e parada antecipada — para otimizar subconjuntos de atributos que maximizam
    o desempenho do modelo fornecido.

    Parâmetros
    ----------
    X : array-like ou pd.DataFrame
        Matriz de dados de entrada (features), tipicamente espectros FTIR ou dados biomédicos.
    y : array-like
        Vetor alvo (classes ou valores contínuos para regressão).
    model : estimador ou objeto com best_estimator_
        Modelo de aprendizado de máquina a ser otimizado.
    generations : int, opcional (default=30)
        Número máximo de gerações do algoritmo genético.
    Tp : int, opcional (default=20)
        Tamanho da população em cada geração.
    gen_random_prob : bool, opcional (default=True)
        Se True, usa probabilidade aleatória para geração dos indivíduos iniciais.
    pairs_number : int, opcional (default=3)
        Número de pares gerados a cada geração para operações de crossover.
    cv : int, opcional (default=5)
        Número de folds para validação cruzada na avaliação de fitness.
    early_stopping : int, opcional (default=10)
        Número de gerações sem melhoria para aplicar parada antecipada.
    insert_self : bool, opcional (default=False)
        Se True, insere uma solução "tudo selecionado" na população inicial.

    Retorno
    -------
    pop_calc : list of tuples
        População final, ordenada por fitness decrescente. Cada elemento é uma tupla (indivíduo, fitness).

    Observações
    -----------
    - O pipeline exibe barras de progresso interativas via Streamlit.
    - Utiliza seleção por roleta (roulette_wheel_selection), crossover simples (simple_crossover) e mutação binária (binary_mutation).
    - Reinserção elitista (sorted_reinsertion) garante sempre os melhores indivíduos.
    - O fitness é avaliado por validação cruzada, de acordo com o modelo fornecido.
    - O processo pode ser interrompido antes caso não haja melhorias após `early_stopping` gerações.

    Exemplo
    -------
    >>> resultado = ag_pipeline(X, y, model, generations=20, Tp=10, cv=3)
    >>> melhor_individuo, melhor_fitness = resultado[0]
    >>> print(f"Melhor fitness: {melhor_fitness:.4f}")

    """

    FTIR_data = X
    my_bar = st.progress(0, text="Running AG: generating initial population...")
    initial_pop = gen_initial_pop(FTIR_data.shape[1], Tp=Tp, random_prob=gen_random_prob)
    if insert_self == True:
        initial_pop.append(np.random.randint(1,2,FTIR_data.shape[1]))    
    my_bar.progress(50, text="Running AG: Calculating fitness of initial population...")
    pop_calc = get_fitness(model, initial_pop, FTIR_data=FTIR_data, y=y, cv=cv)
    my_bar.progress(100, text="Running AG: Calculating fitness of initial population...")
    my_bar.empty()

    no_improves = 0
    my_bar = st.progress(0, text="Running AG generations...")
    for generation_number in range(generations):
        my_bar.progress((generation_number/generations), text=f"Running AG: generation [{generation_number}/{generations}]...")
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
    my_bar.empty()
    return pop_calc