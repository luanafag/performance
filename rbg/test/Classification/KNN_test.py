from rbg.Classification  import KNN
import numpy as np
import pandas as pd
import pytest

class TestDataEntry():
    def test_DataFrame(self):
        X = pd.DataFrame({"coluna1": [1.0, 2.0, 3.0]})
        Y = pd.Series([1, 2, 1], name="coluna2")

        Data = pd.DataFrame({"coluna1": [1.5, 2.0, 0.5]})

        modelo_knn = KNN([1, 2])
        modelo_knn.fit(X, Y)

        DtFrameResult = modelo_knn.predict(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)
        assert DtFrameResult.shape == (3,)
        assert np.isin(DtFrameResult, [1, 2]).all()

    def test_Numpy(self):
        X = np.array([[1.0], [2.0], [3.0]])
        Y = np.array([1, 2, 1])

        Data = np.array([[1.5], [2.0], [0.5]])

        modelo_knn = KNN([1, 2])
        modelo_knn.fit(X, Y)

        NpResult = modelo_knn.predict(Data)

        assert isinstance(NpResult, np.ndarray)
        assert NpResult.shape == (3,)
        assert np.isin(NpResult, [1, 2]).all()

    def test_ListInput(self):
        X = [[1.0], [2.0], [3.0]]
        y = [1, 2, 1]

        Data = [[1.5], [2.5]]

        modelo_knn = KNN([1, 2])
        modelo_knn.fit(X, y)

        Listresult = modelo_knn.predict(Data)

        assert isinstance(Listresult, np.ndarray)
        assert Listresult.shape == (2,)

class TestError():

    def test_MatchingDim(self):
        X = np.zeros((10, 3)) 
        Y = np.zeros(5)       
        
        modelo_knn = KNN([0])

        with pytest.raises(ValueError, match="The data do not match"):
            modelo_knn.fit(X, Y)

        X_cut = X[:5, :] 
        
        modelo_knn.fit(X_cut, Y)

        Data = np.zeros((4, 3))

        resultado = modelo_knn.predict(Data)
        
        assert resultado.shape == (4,)

    def test_params(self):
        modelo_knn = KNN([0])
        modelo_knn.set_params(k = 5)

        di = modelo_knn.get_params()

        assert type(di) == dict
        assert di['k'] == 5 

    def test_invalid_k(self):
        modelo_knn = KNN([0])
        # k = 0 (zero), k negativo, k float, k string, k booleano ou k none
        k_invalid = [0, -1, 2.5, "3", True, None ]
        for k in k_invalid:
            with pytest.raises(ValueError, match="k must be a positive integer."):
               modelo_knn.set_params(k = k)

        X = np.array([[1.0], [2.0]])
        y = np.array([0, 1])

        modelo_knn = KNN([0, 1], k=3)
        modelo_knn.fit(X, y)

        Data = np.array([[1.5]])
        resultado = modelo_knn.predict(Data)

        assert resultado.shape == (1,)
        assert resultado[0] in [0, 1]

    def test_predict(self):
        modelo_knn = KNN([0,1])

        X = np.array([[1.0], [2.0]])
        with pytest.raises(ValueError, match="The model must be trained before prediction."):
            modelo_knn.predict(X)

class TestRightResults():

    def test_correct_prediction(self):
        X = np.array([[1.0], [2.0], [8.0], [9.0]])
        y = np.array([0, 0, 1, 1])

        Data = np.array([[1.5],[8.5],[1.0],[9.0]])

        modelo_knn = KNN([0, 1], k=1)
        modelo_knn.fit(X, y)

        resultado = modelo_knn.predict(Data)
        esperado = np.array([0, 1, 0, 1])

        assert np.array_equal(resultado, esperado)

    def test_multiple_features(self):
        X = np.array([[1.0, 1.0],[2.0, 2.0],[8.0, 8.0],[9.0, 9.0]])
        y = np.array([0, 0, 1, 1])

        Data = np.array([[1.5, 1.5],[8.5, 8.5]])

        modelo_knn = KNN([0, 1], k=1)
        modelo_knn.fit(X, y)

        resultado = modelo_knn.predict(Data)
        esperado = np.array([0, 1])

        assert np.array_equal(resultado, esperado)

    def test_one_training(self):
        X = np.array([[5.0]])
        y = np.array([1])

        Data = np.array([[5.0], [6.0]])

        modelo_knn = KNN([0, 1], k=1)
        modelo_knn.fit(X, y)

        resultado = modelo_knn.predict(Data)
        esperado = np.array([1, 1])

        assert np.array_equal(resultado, esperado)

    def test_draw(self):
        X = np.array([[1.0], [3.0]])
        y = np.array([0, 1])

        Data = np.array([[2.0]])

        modelo_knn = KNN([0, 1], k=2)
        modelo_knn.fit(X, y)

        resultado = modelo_knn.predict(Data)
        esperado = np.array([0])

        assert np.array_equal(resultado, esperado)