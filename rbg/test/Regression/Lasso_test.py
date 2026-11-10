from rbg.Regression.Lasso import Lasso
import numpy as np
import pandas as pd
import pytest

"""
#verificar apenas se aceita receber o parametro alpha
Lasso.set_params(alpha = 1,intercept = False)
Lasso.get_params()

#Testa se recebe o que deve receber e se retorna o tipo de dado que deve receber, testa se retorna erro para dados de tamanhos diferentes
Lasso.fit(X,Y)

#Testa se retorna uma array numpy de tamanho compativel
Lasso.predict(Data)

#Testa se retorna oma array numpy com coeficientes
Lasso.coef

"""


class TestDataEntry():
    def test_DataFrame(self):
        X = pd.DataFrame({"coluna1": [1.0, 2.0, 3.0], "coluna2": [2.0, 3.0, 4.0]})
        Y = pd.Series([10.5, 20.5, 30.5]) 

        Data = pd.DataFrame({"coluna1": [1.5, 2.0, 0.5], "coluna2": [2.5, 3.0, 1.5]})

        modelo_lasso = Lasso()
        modelo_lasso.fit(X, Y)

        DtFrameResult = modelo_lasso.predict(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)

    def test_Numpy(self):
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        Y = np.array([10.5, 20.5, 30.5])

        Data = np.array([[1.5, 2.5], [2.0, 3.0], [0.5, 1.5]])

        modelo_lasso = Lasso()
        modelo_lasso.fit(X, Y)

        NpResult = modelo_lasso.predict(Data)

        assert isinstance(NpResult, np.ndarray)

class TestError():
    def test_MatchingDim(self):
        X = np.zeros((10, 3)) 
        Y = np.zeros(5)       
        
        modelo_lasso = Lasso()

        with pytest.raises(ValueError, match="same number of observations"):
            modelo_lasso.fit(X, Y)

        X_cut = X[:5, :] 
        
        modelo_lasso.fit(X_cut, Y)

        Data = np.zeros((4, 3))

        resultado = modelo_lasso.predict(Data)
        
        assert isinstance(resultado, np.ndarray)
        assert resultado.shape == (4,)

    def test_params(self):
        modelo_lasso = Lasso()
        modelo_lasso.set_params(alpha=1, fit_intercept=False)
        parametros = modelo_lasso.get_params()
        
        assert parametros is not None

    def test_attributes(self):
        xtrain = np.array([
            [1.5, 2.0], 
            [1.6, 1.9], 
            [8.5, 9.0], 
            [8.6, 9.1]
        ])
        ytrain = np.array([10.0, 12.0, 20.0, 22.0])
        
        n_features = xtrain.shape[1]
        
        modelo_lasso = Lasso()
        modelo_lasso.fit(xtrain, ytrain)

        assert isinstance(modelo_lasso.coef, np.ndarray)
        assert modelo_lasso.coef.shape == (n_features + 1,)

#============fazer
class TestRightResults():
    pass
