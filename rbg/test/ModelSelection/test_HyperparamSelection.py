'''
GridSearch.set_params(modelo = modelo_generico, hyperparametros = {parametro1 = [1,2,3,4] , parametro2 = ['1','2']}, tecnica_de_particao = Kfold, metrics = MSE)
GridSearch

#adiciona os dados na classe
Hyperparams.fit(X,Y)

#a matriz retornaria os melhores hyperparametros
melhor_hyperparametro = GridSearch.best_hyperparams


#aqui a matriz guardaria o valor da metrica retornado de cada seleção de parametros
matriz = GridSearch.matrix_value

'''

import pytest
import numpy as np
from rbg.ModelSelection import GridSearch, KFold, LOOCV
from rbg.Utilities import MSE, Resample, Accuracy # devia ser accuracy
from rbg.Classification import KNN


def test_HyperparamSelection():
    hiperparametros_injetados = {'k': [1, 2, 3, 4]}
    
    X_treino = np.array([
        [1.0, 2.0], 
        [2.0, 4.0], 
        [3.0, 6.0], 
        [4.0, 8.0]
    ])
    Y_treino = np.array(['azul','azul', 'amarelo', 'vermelho'])

    #inicializando a classe
    seletor = GridSearch(
        model=KNN(labels = ['azul','amarelo','vermelho']), 
        param_grid=hiperparametros_injetados, 
        cv=KFold(k=2), 
        metrics=Accuracy,
        greater_is_better=True
    )
    
    # parametros
    novos_parametros = {'k': [100, 200]}
    seletor.set_params(hyperparameters=novos_parametros)
    
    params_resgatados = seletor.get_params()
    
    # testes
    assert isinstance(params_resgatados, dict)
    assert params_resgatados.get('k') == [100, 200]

    seletor.set_params(hyperparameters=hiperparametros_injetados)
    
    
    seletor.fit(X_treino, Y_treino)
    
    # ===================== MELHOR HYPER PARAMETRO ==============================
    melhor_hyperparametro = seletor.best_params_
    
    #tipo de dado
    assert isinstance(melhor_hyperparametro, dict)

    #tamanho
    assert len(melhor_hyperparametro) == len(hiperparametros_injetados)

    #todas as chaves foram selecionadas
    for chave in hiperparametros_injetados.keys():
        assert chave in melhor_hyperparametro


"""    # ============= MATRIZ ====================
    matriz = seletor.matrix_value
    
    assert isinstance(matriz, np.ndarray)
    
    tamanho_param1 = len(hiperparametros_injetados['parametro1']) 
    tamanho_param2 = len(hiperparametros_injetados['parametro2']) 
    assert matriz.shape == (tamanho_param1, tamanho_param2)

    
    score_combinacao_inicial = matriz[0, 0]
    
    assert isinstance(score_combinacao_inicial, (float, np.number))
"""
#100th commit 