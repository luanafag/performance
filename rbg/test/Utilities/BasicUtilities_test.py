# from rbg.Utilities import matrix_design, Resample
from rbg.Utilities import Resample
from rbg.Utilities import CoordinateDescent
from rbg.Utilities import MatrixDesign


'''
#a partir de uma array constroi uma matriz design
matrix_design(X, intecept = True)


#Bootsratp
Resample(data,size = len(data),qte = 10 ,replace = True,)
#deve retornar uam array com tamanho qte em que cada indice representa uma nova reamostragem
'''


import numpy as np
import pytest



class TestBootstrap:
    
    @pytest.mark.parametrize("data, size, qte, replace", [
        #parametros pro teste
        (np.array([1, 2, 3, 4, 5]), 5, 10, True),
        (np.array([1, 2, 3, 4, 5, 6, 7, 8]), 3, 5, True),
        (np.array([10, 20, 30, 40]), 4, 2, False),        
        (np.array([7, 8, 9]), 1, 3, True) # esse q ta dando problema?
    ])
    def test_Resample_parametrizado(self, data, size, qte, replace):
        
        resultado = Resample(X = data, size=size, n_resamples=qte, replace=replace)
        
        assert isinstance(resultado, (np.ndarray, list))
        
        assert len(resultado) == qte
        
        for amostra in resultado:
            assert len(amostra) == 2 # retorna treino e teste
            assert len(amostra[0]) == size ## indices de treino tem tamanho certo


class TestCoordinateDescent:
    def test_l1_shrinkage(self):
        """Check a regular solution."""
        X = np.eye(3)
        y = np.array([3.0, 0.5, -2.0])

        beta = CoordinateDescent.solve(X, y, 1.0, False)

        assert isinstance(beta, np.ndarray)
        assert beta.shape == (3,)
        np.testing.assert_allclose(beta, [2.0, 0.0, -1.0], atol=1e-5)


    def test_unpenalized_intercept(self):
        """Intercept test"""
        X = np.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])
        y = np.array([1.0, 4.0, 7.0])

        beta = CoordinateDescent.solve(X, y, 2.0, True)

        assert isinstance(beta, np.ndarray)
        assert beta.shape == (2,)
        np.testing.assert_allclose(beta, [4.0, 2.0], atol=1e-5)



class TestMatrixdesign:
    def test_dimensoes_e_parametros(self):
            X = np.array([
                [2.0, 3.0], 
                [4.0, 5.0],
                [6.0, 7.0]
            ])
            n_amostras = X.shape[0]
            n_features = X.shape[1]
            
            resultado_base = MatrixDesign(X, degree=[2,2], intercept=False)
            assert isinstance(resultado_base, np.ndarray)
            assert resultado_base.shape == (n_amostras, n_features*2)
            
            resultado_intercept = MatrixDesign(X, degree=[2,3], intercept=True)
            assert resultado_intercept.shape == (n_amostras, 6)
            