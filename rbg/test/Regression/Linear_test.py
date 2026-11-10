from rbg.Regression.Linear import Linear
import numpy as np
import pandas as pd
import pytest

'''
Linear.set_params(intecept = False)
Linear.get_params()

#Testa se recebe o que deve receber e se retorna o tipo de dado que deve receber, testa se retorna erro para dados de tamanhos diferentes
Linear.fit(X,Y)

#Testa se retorna uma array numpy de tamanho compativel
Linear.predict(Data)

#Testa se retorna oma array numpy com coeficientes
Linear.coef
'''


class TestDataEntry():
    def test_DataFrame(self):
        X = pd.DataFrame({"coluna1": [1.0, 2.0, 3.0], "coluna2": [2.0, 3.0, 4.0]})
        Y = pd.Series([10.5, 20.5, 30.5]) 

        Data = pd.DataFrame({"coluna1": [1.5, 2.0, 0.5], "coluna2": [2.5, 3.0, 1.5]})

        modelo_linear = Linear()
        modelo_linear.fit(X, Y)

        DtFrameResult = modelo_linear.predict(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)

    def test_Numpy(self):
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        Y = np.array([10.5, 20.5, 30.5])

        Data = np.array([[1.5, 2.5], [2.0, 3.0], [0.5, 1.5]])

        modelo_linear = Linear()
        modelo_linear.fit(X, Y)

        NpResult = modelo_linear.predict(Data)

        assert isinstance(NpResult, np.ndarray)

class TestError():
    def test_MatchingDim(self):
        X = np.zeros((10, 3)) 
        Y = np.zeros(5)       
        
        modelo_linear = Linear()

        with pytest.raises(ValueError, match="same number of observations"):
            modelo_linear.fit(X, Y)

        X_cut = X[:5, :] 
        
        modelo_linear.fit(X_cut, Y)

        Data = np.zeros((4, 3))

        resultado = modelo_linear.predict(Data)
        
        assert isinstance(resultado, np.ndarray)
        assert resultado.shape == (4,)

    def test_params(self):
        modelo_linear = Linear()
        modelo_linear.set_params(fit_intercept=False)
        parametros = modelo_linear.get_params()
        
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
        
        modelo_linear = Linear()
        modelo_linear.fit(xtrain, ytrain)

        assert isinstance(modelo_linear.coef, np.ndarray)
        assert modelo_linear.coef.shape == (n_features + 1,)
        
#============= Fazer ainda =================
class TestRightResults():
    pass
