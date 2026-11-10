
from rbg.Classification import QDA as QDA
import numpy as np
import pandas as pd
import pytest


class TestDataEntry():
    def test_DataFrame(self):
        X = pd.DataFrame({"coluna1": [1.0, 2.0, 3.0, 2.5]})
        Y = pd.Series(['contaminated', 'drinkable', 'contaminated', 'drinkable'], name="coluna2")

        Data = pd.DataFrame({"coluna1": [1.5, 2.0, 0.5, 2.5]})

        modelo_qda = QDA(["contaminated", "drinkable"])
        modelo_qda.fit(X, Y)

        DtFrameResult = modelo_qda.predict(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)
        assert DtFrameResult.shape == (4,)
        assert np.isin(DtFrameResult, ["contaminated", "drinkable"]).all()

    def test_Numpy(self):
        X = np.array([[1.0], [2.0], [3.0], [2.5]])
        Y = np.array(['contaminated', 'drinkable', 'contaminated', 'drinkable'])

        Data = np.array([[1.5], [2.0], [0.5], [2.5]])

        modelo_qda = QDA(["contaminated", "drinkable"])
        modelo_qda.fit(X, Y)

        NpResult = modelo_qda.predict(Data)

        assert isinstance(NpResult, np.ndarray)
        assert NpResult.shape == (4,)
        assert np.isin(NpResult, ["contaminated", "drinkable"]).all()

    def test_ListInput(self):
        X = [[1.0], [2.0], [3.0], [2.5]]
        y = [0, 1, 0, 1]

        Data = [[1.5], [2.0], [0.5]]

        modelo_qda = QDA([0, 1])
        modelo_qda.fit(X, y)

        Listresult = modelo_qda.predict(Data)

        assert isinstance(Listresult, np.ndarray)
        assert Listresult.shape == (3,)


class TestError():

    def test_MatchingDim(self):
        X = np.zeros((10, 3))
        Y = np.zeros(5)
                
        modelo_qda = QDA([0,1])
        with pytest.raises(ValueError, match="The data do not match"):
            modelo_qda.fit(X, Y)
        
        # valid train set
        X_treino = np.array([[0.0, 0.0],[0.1, -0.1],[4.0, 4.0],[4.1, 4.1], [0.2, 0.6], [3.5, 4]])
        Y_treino = np.array([0, 0, 1, 1, 0, 1])
        modelo_qda.fit(X_treino, Y_treino)
                
        # prediction verification
        Data = np.array([[0.05, -0.05],[4.05, 4.05],[0.02, -0.02],[4.08, 4.08]])
        
        resultado = modelo_qda.predict(Data)
        
        assert isinstance(resultado, np.ndarray)
        assert resultado.shape == (4,)

    def test_params(self):
        modelo_qda = QDA(["contaminated", "drinkable"], reference_class="drinkable")

        assert np.array_equal(modelo_qda._labels, np.array(["contaminated", "drinkable"]))
        assert np.array_equal(modelo_qda.labels, np.array(["contaminated", "drinkable"]))
        assert modelo_qda._reference_class == "drinkable"

    def test_attributes(self):

        X_train = np.array([[1.5, 2.0],[1.6, 1.9],[8.5, 9.0],[8.6, 9.1],[2.0, 1.5],[9.5, 8.5],[1.0, 1.5],[7.5, 9.0]])
        Y_train = np.array(['contaminated', 'contaminated', 'drinkable', 'drinkable', 'contaminated', 'drinkable', 'contaminated', 'drinkable'])
        
        modelo_qda = QDA(["contaminated", "drinkable"])
        modelo_qda.fit(X_train, Y_train)
      
        assert isinstance(modelo_qda.labels, np.ndarray)
        assert np.array_equal(np.unique(modelo_qda.labels), np.unique(Y_train))
    

    def test_method(self):
        X_train = np.array([[1.5, 2.0],[1.6, 1.9],[8.5, 9.0],[8.6, 9.1],[2.0, 1.5],[9.5, 8.5],[1.0, 1.5],[7.5, 9.0]])
        Y_train = np.array(['contaminated', 'contaminated', 'drinkable', 'drinkable', 'contaminated', 'drinkable', 'contaminated', 'drinkable'])
        
        modelo_qda = QDA(["contaminated", "drinkable"])
        modelo_qda.fit(X_train, Y_train)

        label = modelo_qda.labels[0]
        n_features = X_train.shape[1] 

        cov_matrix = modelo_qda.cov_matrix(k=label)
        assert isinstance(cov_matrix, np.ndarray)
        assert cov_matrix.shape == (n_features, n_features)

        mean_array = modelo_qda.mean_array(k=label)
        assert isinstance(mean_array, np.ndarray)
        assert mean_array.shape == (n_features,)

        prior = modelo_qda.prior(k=label)
        assert isinstance(prior, (float, np.number))

        k_function = modelo_qda.k_functions(k=label)
        assert callable(k_function)

        amostra_teste = X_train[0]
        resultado_func = k_function(amostra_teste)
        
        assert isinstance(resultado_func, (float, np.number))

    def test_fit_train(self):
        modelo_qda = QDA([0, 1])

        # X one dimension
        X = np.array([1, 2, 3])
        y = np.array([0, 0, 1])
        with pytest.raises(ValueError, match="X must be two-dimensional."):
            modelo_qda.fit(X, y)

        # X three dimension
        X = np.zeros((2, 2, 3))
        y = np.array([0, 1])
        with pytest.raises(ValueError, match="X must be two-dimensional."):
            modelo_qda.fit(X, y)

        # X empty
        X = np.zeros((0, 1))
        y = np.array([])
        with pytest.raises(ValueError, match="X must not be empty."):
            modelo_qda.fit(X, y)

        # X no features
        X = np.zeros((4, 0))
        y = np.array([0, 0, 1, 1])
        with pytest.raises(ValueError, match="X must have at least one feature."):
            modelo_qda.fit(X, y)

        # y two dimensions
        X = np.array([[1.0], [2.0], [8.0], [9.0]])
        y = np.array([[0], [0], [1], [1]])
        with pytest.raises(ValueError, match="y must be one-dimensional."):
            modelo_qda.fit(X, y)

        # X nan, inf or -inf
        X_invalid = [np.array([[1.0], [np.nan]]), np.array([[1.0], [np.inf]]), np.array([[1.0], [-np.inf]])]
        for X in X_invalid:
            with pytest.raises(ValueError, match="X must have only finite values."):
                modelo_qda.fit(X, np.array([0, 1]))

        # y com classes desconhecidas
        X = np.array([[1.0], [2.0], [3.0], [8.0]])
        y = np.array([0, 1, 2, 1])
        with pytest.raises(ValueError, match="y contains unknown class labels."):
            modelo_qda.fit(X, y)

        # X com strings
        X = np.array([["a"], ["b"], ["c"], ["d"]])
        y = np.array([0, 0, 1, 1])
        with pytest.raises(ValueError, match="X must be a numerical array."):
            modelo_qda.fit(X, y)

        # Apenas uma classe
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 0, 0])
        with pytest.raises(ValueError, match="y must be at least two classes."):
            modelo_qda.fit(X, y)

        # Classes com menos de duas amostras
        X = np.array([[1.0], [2.0], [8.0]])
        y = np.array([0, 0, 1])
        with pytest.raises(ValueError, match="Each class must have at least two samples."):
            modelo_qda.fit(X, y)

        # Covariance matrix singular
        X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [8.0, 8.0], [9.0, 9.0], [10.0, 10.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        with pytest.raises(ValueError, match="The covariance matrix is singular."):
            modelo_qda.fit(X, y)

    def test_predict(self):
        modelo_qda = QDA([0, 1])

        # not trained model
        X = np.array([[1.0], [2.0]])
        with pytest.raises(ValueError, match="The model must be trained before prediction."):
            modelo_qda.predict(X)

        # valid train
        X_train = np.array([[1.0], [2.0], [3.0], [8.0], [9.0], [10.0]])
        y_train = np.array([0, 0, 0, 1, 1, 1])
        modelo_qda.fit(X_train, y_train)

        # X one dimension
        X = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="X must be two-dimensional."):
            modelo_qda.predict(X)

        # X three dimension
        X = np.zeros((1, 2, 3))
        with pytest.raises(ValueError, match="X must be two-dimensional."):
            modelo_qda.predict(X)

        # X empty
        X = np.empty((0, 1))
        with pytest.raises(ValueError, match="X must not be empty."):
            modelo_qda.predict(X)

        # wrong features
        X = np.zeros((2, 3))
        with pytest.raises(ValueError, match="X must be the same number of features as the training X."):
            modelo_qda.predict(X)

        # X nan, inf or -inf
        X_invalid = [np.array([[1.0], [np.nan]]), np.array([[1.0], [np.inf]]), np.array([[1.0], [-np.inf]])]
        for X in X_invalid:
            with pytest.raises(ValueError, match="X must have only finite values."):
                modelo_qda.predict(X)

        # X com strings
        X = np.array([["a"], ["b"]])
        with pytest.raises(ValueError, match="X must be a numerical array."):
            modelo_qda.predict(X)


class TestRightResults():

    def test_correct_prediction(self):
        X = np.array([[1.0], [2.0], [3.0], [8.0], [9.0], [10.0]])
        y = np.array([0, 0, 0, 1, 1, 1])

        Data = np.array([[1.5], [8.5], [1.0], [9.0]])

        modelo_qda = QDA([0, 1])
        modelo_qda.fit(X, y)

        resultado = modelo_qda.predict(Data)
        esperado = np.array([0, 1, 0, 1])

        assert np.array_equal(resultado, esperado)

    def test_multiple_features(self):
       X = np.array([[1.0, 1.0], [2.0, 2.5], [3.0, 1.5], [8.0, 8.0], [9.0, 9.5], [10.0, 8.5]])
       y = np.array([0, 0, 0, 1, 1, 1])

       Data = np.array([[1.5, 1.5], [8.5, 8.5]])

       modelo_qda = QDA([0, 1])
       modelo_qda.fit(X, y)

       resultado = modelo_qda.predict(Data)
       esperado = np.array([0, 1])

       assert np.array_equal(resultado, esperado)


    def test_one_training(self):
        X = np.array([[1.0], [2.0], [3.0], [8.0], [9.0], [10.0]])
        y = np.array([0, 0, 0, 1, 1, 1])

        Data = np.array([[1.5], [2.0]])

        modelo_qda = QDA([0, 1])
        modelo_qda.fit(X, y)

        resultado = modelo_qda.predict(Data)
        esperado = np.array([0, 0])

        assert np.array_equal(resultado, esperado)

    def test_draw(self):
        X = np.array([[1.0], [2.0], [3.0], [7.0], [8.0], [9.0]])
        y = np.array([0, 0, 0, 1, 1, 1])

        Data = np.array([[5.0]])

        modelo_qda = QDA([0, 1])
        modelo_qda.fit(X, y)

        resultado = modelo_qda.predict(Data)
        esperado = np.array([0])

        assert np.array_equal(resultado, esperado)
