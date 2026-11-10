"""Check exact polynomial expansions and dimensional contracts."""
import numpy as np
import pytest
from rbg.Regression.Polynomial import Polynomial

''' Guia dos testes basicos
Poly = Polynomial()
Poly.set_params(intecept = False)
Poly.get_params()

#Testa se recebe o que deve receber e se retorna o tipo de dado que deve receber, testa se retorna erro para dados de tamanhos diferentes
Poly.fit(X,Y)

#Testa se retorna uma array numpy de tamanho compativel
Poly.predict(Data)

#Testa se retorna oma array numpy com coeficientes
Poly.coef
'''

class TestDataEntry:
    def test_Numpy(self):
        X = np.array([
            [1.0, 2.0], 
            [2.0, 1.5], 
            [3.0, 4.0],
            [4.5, 2.1],
            [5.0, 3.3],
            [1.1, 7.0]
        ])
        Y = np.array([10.5, 12.0, 30.5, 22.1, 35.0, 40.0])

        Data = np.array([[1.5, 2.5], [2.0, 3.0], [0.5, 1.5]])

        modelo_poly = Polynomial(degree=[2, 2])
        modelo_poly.fit(X, Y)

        NpResult = modelo_poly.predict(Data)
        assert isinstance(NpResult, np.ndarray)


class TestError:
    def test_MatchingDim(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(15, 3)) 
        Y = rng.normal(size=(8,))       
        
        modelo_poly = Polynomial(degree=[2, 2, 2])

        # testa se recusa shapes diferentes 
        with pytest.raises(ValueError, match="same number of observations"):
            modelo_poly.fit(X, Y)

        X_cut = X[:8, :] 
        
        modelo_poly.fit(X_cut, Y)

        Data = rng.normal(size=(4, 3))

        resultado = modelo_poly.predict(Data)
        
        assert isinstance(resultado, np.ndarray)
        assert resultado.shape == (4,)

    def test_params(self):
        modelo_poly = Polynomial(degree=[2, 2])
        modelo_poly.set_params(fit_intercept=False)
        parametros = modelo_poly.get_params()
        
        assert parametros is not None

    def test_attributes(self):
        xtrain = np.array([
            [1.5, 2.0], 
            [1.6, 1.9], 
            [8.5, 9.0], 
            [8.6, 9.1],
            [3.0, 4.0],
            [5.5, 2.2]
        ])
        ytrain = np.array([10.0, 12.0, 20.0, 22.0, 15.0, 18.0])
        
        modelo_poly = Polynomial(degree=[2, 2])
        modelo_poly.fit(xtrain, ytrain)

        # testa se retorna uma array numpy com coeficientes
        assert isinstance(modelo_poly.coef, np.ndarray)
        assert len(modelo_poly.coef) > 0
        assert modelo_poly.coef.ndim == 1

class TestPolynomialTransform:

    @pytest.fixture(autouse=True)
    def _subtest_context(self, subtests):
        self.subtests = subtests

    def test_transform_known_combinations(self) -> None:
        cases = [
            ([1], [[2], [-3], [0]], [[2], [-3], [0]]),
            ([3], [[2], [-2], [0]], [[2, 4, 8], [-2, 4, -8], [0, 0, 0]]),
            ([2, 3], [[2, 3]], [[2, 4, 3, 9, 27]]),
            ([2, 1], [[-2, 0], [0.5, -3]], [[-2, 4, 0], [0.5, 0.25, -3]]),
            ([1, 2, 1], [[1, -2, 3], [0, 0, -1]], [[1, -2, 4, 3], [0, 0, 0, -1]]),
            ([2, 2], [[1, 2], [3, 4]], [[1, 1, 2, 4], [3, 9, 4, 16]]),
        ]
        for degrees, X, expected in cases:
            for intercept in (True, False):
                with self.subtests.test(degrees=degrees, X=X, intercept=intercept):
                    original = np.array(X, dtype=float)
                    result = Polynomial(degree=degrees, fit_intercept=intercept)._transform(original)
                    assert result.shape == (len(X), sum(degrees))
                    np.testing.assert_array_equal(result, expected)
                    np.testing.assert_array_equal(original, X)

    def test_transform_empty_batch_preserves_columns(self) -> None:
        result = Polynomial(degree=[2, 3])._transform(np.empty((0, 2)))
        assert result.shape == (0, 5)

    def test_transform_rejects_wrong_dimensions(self) -> None:
        model = Polynomial(degree=[2, 2])
        for X in (np.array(2), np.array([2, 3]), np.ones((2, 2, 1)),
                  np.ones((3, 1)), np.ones((3, 3))):
            with self.subtests.test(shape=X.shape):
                with pytest.raises(ValueError):
                    model._transform(X)

    def test_polynomial_function_dimensions(self) -> None:
        rng = np.random.default_rng(123)
        X = rng.normal(size=(24, 2))
        y = 1 + 2 * X[:, 0] + X[:, 0]**2 - X[:, 1]
        for method in ("linear", "ridge"):
            for intercept in (True, False):
                with self.subtests.test(method=method, intercept=intercept):
                    model = Polynomial(degree=[2, 1], method=method, fit_intercept=intercept)
                    p = 3 + int(intercept)
                    options = {"alpha": 0.1} if method == "ridge" else {}
                    beta = model.fit(X, y, **options)
                    assert beta.shape == (p,)
                    for n in (1, 5):
                        assert model.predict(X[:n]).shape == (n,)
                    assert model.residuals().shape == (len(X),)
                    inference_beta, variance, covariance, residuals = model.inference()
                    assert inference_beta.shape == (p,)
                    assert np.ndim(variance) == 0
                    assert covariance.shape == (p, p)
                    assert residuals.shape == (len(X),)
