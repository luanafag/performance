'''
#erro absoluto
MAE(Y_predict,Y_real)

#raiz quadrada do mse
RMSE(Y_predict,Y_real)

#recebe dois pontos e retorna a soma da diferença entre cada par respectivo ao quadrado dividdida pela quantidade de valores, 
MSE(Y_predict,Y_real)

#R² coeficiente de determinacao
RSquare(Y_predict,Y_real)
'''

import pytest
import numpy as np
from rbg.Utilities import MAE,RMSE,MSE,RSquare

class TestRegressionMetrics:
    
    def test_MAE(self):
        Y_real = np.array([3.0, -0.5, 2.0, 7.0])
        Y_predict = np.array([2.5, 0.0, 2.0, 8.0])
        
        resultado = MAE(Y_predict, Y_real)
        
        assert isinstance(resultado, (float, np.number))
        assert resultado >= 0.0

    def test_RMSE(self):
        Y_real = np.array([3.0, -0.5, 2.0, 7.0])
        Y_predict = np.array([2.5, 0.0, 2.0, 8.0])
        
        resultado = RMSE(Y_predict, Y_real)
        
        assert isinstance(resultado, (float, np.number))
        assert resultado >= 0.0

    def test_MSE(self):
        Y_real = np.array([3.0, -0.5, 2.0, 7.0])
        Y_predict = np.array([2.5, 0.0, 2.0, 8.0])
        
        resultado = MSE(Y_predict, Y_real)
        
        assert isinstance(resultado, (float, np.number))
        assert resultado >= 0.0

    def test_RSquare(self):
        Y_real = np.array([3.0, -0.5, 2.0, 7.0])
        Y_predict = np.array([2.5, 0.0, 2.0, 8.0])
        
        resultado = RSquare(Y_predict, Y_real)
        
        assert isinstance(resultado, (float, np.number))
        assert resultado <= 1.0
