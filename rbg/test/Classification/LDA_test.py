from rbg.Classification import LDA
import numpy as np
import pandas as pd
import pytest


class TestDataEntry():
    def test_DataFrame(self):
        X = pd.DataFrame({"coluna1": [1.0, 2.0, 3.0, 8.0, 9.0, 10.0]})
        Y = pd.Series(['contaminated', 'drinkable', 'contaminated', 'drinkable', 'contaminated', 'drinkable'], name="coluna2")

        Data = pd.DataFrame({"coluna1": [1.5, 2.0, 0.5]})

        modelo_lda = LDA(labels = ["contaminated", "drinkable"])
        modelo_lda.fit(X, Y)

        DtFrameResult = modelo_lda.predict(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)
        assert DtFrameResult.shape == (3,)
        assert np.isin(DtFrameResult, ["contaminated", "drinkable"]).all()

    def test_Numpy(self):
        X = np.array([[1.0], [2.0], [3.0], [8.0], [9.0], [10.0]])
        Y = np.array(['contaminated', 'drinkable', 'contaminated', 'drinkable', 'contaminated', 'drinkable'])

        Data = np.array([[1.5], [2.0], [0.5]])

        modelo_lda = LDA(labels  = ["contaminated", "drinkable"])
        modelo_lda.fit(X, Y)

        NpResult = modelo_lda.predict(Data)

        assert isinstance(NpResult, np.ndarray)
        assert NpResult.shape == (3,)
        assert np.isin(NpResult, ["contaminated", "drinkable"]).all()

    def test_ListInput(self):
        X = [[1.0], [2.0], [3.0], [8.0], [9.0], [10.0]]
        y = [0, 0, 0, 1, 1, 1]

        Data = [[1.5], [2.5]]

        modelo_lda = LDA([0, 1])
        modelo_lda.fit(X, y)

        Listresult = modelo_lda.predict(Data)

        assert isinstance(Listresult, np.ndarray)
        assert Listresult.shape == (2,)

class TestError():

    def test_MatchingDim(self):
        X = np.zeros((10, 3))
        Y = np.zeros(5)

        modelo_lda = LDA(labels =[0,1])

        with pytest.raises(ValueError, match="The data do not match"):
            modelo_lda.fit(X, Y)

        # valid train set
        X_cut = np.array([[0.0, 0.0], [0.1, -0.1], [4.0, 4.0], [4.1, 4.1], [0.2, 0.1], [4.2, 4.1]])
        Y_cut = np.array([0, 0, 1, 1, 0, 1])

        modelo_lda.fit(X_cut, Y_cut)

        # prediction verification
        Data = np.array([[0.05, -0.05], [4.05, 4.05], [0.02, -0.02], [4.08, 4.08]])

        resultado = modelo_lda.predict(Data)

        assert isinstance(resultado, np.ndarray)
        assert resultado.shape == (4,)

    def test_params(self):
        modelo_lda = LDA(["contaminated", "drinkable"])

        assert np.array_equal(modelo_lda._labels, np.array(["drinkable", "contaminated"]))
        assert np.array_equal(modelo_lda.labels, np.array(["contaminated", "drinkable"]))
        assert modelo_lda._reference_class == "contaminated"

    def test_attributes(self):
        X_treino = np.array([[1.5, 2.0], [1.6, 1.9], [8.5, 9.0], [8.6, 9.1], [1.7, 2.1], [8.7, 9.2]])
        Y_treino = np.array(['contaminated', 'contaminated', 'drinkable', 'drinkable', 'contaminated', 'drinkable'])

        modelo_lda = LDA(labels =["contaminated", "drinkable"])
        modelo_lda.fit(X_treino, Y_treino)

        assert isinstance(modelo_lda.labels, np.ndarray)
        assert np.array_equal(np.unique(modelo_lda.labels), np.unique(Y_treino))
        assert np.array_equal(modelo_lda.labels, np.array(['contaminated', 'drinkable']))

    def test_method(self):

        X_treino = np.array([[1.5, 2.0], [1.6, 1.9], [8.5, 9.0], [8.6, 9.1], [1.7, 2.1], [8.7, 9.2]])
        Y_treino = np.array([0, 0, 1, 1, 0, 1])

        n_features = X_treino.shape[1]

        modelo_lda = LDA([0,1])
        modelo_lda.fit(X_treino, Y_treino)

        cov_matrix = modelo_lda.cov_matrix()
        assert isinstance(cov_matrix, np.ndarray)
        assert cov_matrix.shape == (n_features, n_features)

        mean_array = modelo_lda.mean_array(k = modelo_lda.labels[0])
        assert isinstance(mean_array, np.ndarray)
        assert mean_array.shape == (n_features,)

        prior = modelo_lda.prior(k = modelo_lda.labels[0])
        assert isinstance(prior, np.number)

        k_function = modelo_lda.k_functions(k = modelo_lda.labels[0])

        assert callable(k_function)

        amostra_teste = X_treino[0]
        resultado_func = k_function(amostra_teste)

        assert isinstance(resultado_func, (float, np.number))

class TestRightResults():

    def test_correct_prediction(self):
        X = np.array([[1.0], [2.0], [3.0], [8.0], [9.0], [10.0]])
        y = np.array([0, 0, 0, 1, 1, 1])

        Data = np.array([[1.5], [8.5], [1.0], [9.0]])

        modelo_lda = LDA([0, 1])
        modelo_lda.fit(X, y)

        resultado = modelo_lda.predict(Data)
        esperado = np.array([0, 1, 0, 1])

        assert np.array_equal(resultado, esperado)

    def test_multiple_features(self):
        X = np.array([[1.0, 1.0], [2.0, 2.1], [3.0, 2.9], [8.0, 8.1], [9.0, 9.0], [10.0, 9.9]])
        y = np.array([0, 0, 0, 1, 1, 1])

        Data = np.array([[1.5, 1.5], [8.5, 8.5]])

        modelo_lda = LDA([0, 1])
        modelo_lda.fit(X, y)

        resultado = modelo_lda.predict(Data)
        esperado = np.array([0, 1])

        assert np.array_equal(resultado, esperado)

    def test_one_training(self):
        X = np.array([[1.0], [2.0], [3.0], [8.0], [9.0], [10.0]])
        y = np.array([0, 0, 0, 1, 1, 1])

        Data = np.array([[1.5], [2.0]])

        modelo_lda = LDA([0, 1])
        modelo_lda.fit(X, y)

        resultado = modelo_lda.predict(Data)
        esperado = np.array([0, 0])

        assert np.array_equal(resultado, esperado)

    def test_draw(self):
        X = np.array([[1.0], [2.0], [3.0], [8.0], [9.0], [10.0]])
        y = np.array([0, 0, 0, 1, 1, 1])

        Data = np.array([[5.5]])

        modelo_lda = LDA([0, 1])
        modelo_lda.fit(X, y)

        resultado = modelo_lda.predict(Data)
        esperado = np.array([0])

        assert np.array_equal(resultado, esperado)
