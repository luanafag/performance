from rbg.Unsupervised.PCA import PCA
import numpy as np
import pandas as pd
import pytest


class TestDataEntry():
    def test_DataFrame(self):
        pca = PCA()
        X = pd.DataFrame({"coluna1": [1.0, 2.0, 3.0, 4.0], "coluna2": [2.0, 1.5, 3.5, 4.0]})

        Data = pd.DataFrame({"coluna1": [1.5, 2.0, 0.5], "coluna2": [2.5, 3.0, 1.5]})

        pca.set_params(n_components=1)
        pca.fit(X)

        DtFrameResult = pca.transform(Data)

        assert isinstance(DtFrameResult, np.ndarray)

    def test_Numpy(self):
        pca = PCA()
        X = np.array([[1.0, 2.0], [2.0, 1.5], [3.0, 3.5], [4.0, 4.0]])

        Data = np.array([[1.5, 2.5], [2.0, 3.0], [0.5, 1.5]])

        pca.set_params(n_components=1)
        pca.fit(X)

        NpResult = pca.transform(Data)

        assert isinstance(NpResult, np.ndarray)


class TestError():
    def test_MatchingDim(self):
        pca = PCA()
        X = np.array([
            [1.5, 2.0, 0.3],
            [1.6, 1.9, 0.8],
            [8.5, 9.0, 4.1],
            [8.6, 9.1, 3.7],
            [4.2, 5.3, 2.2]
        ])

        pca.set_params(n_components=2)
        pca.fit(X)

        with pytest.raises(ValueError, match="The data do not match"):
            pca.transform(np.zeros((4, 2)))

        resultado = pca.transform(np.zeros((4, 3)))

        assert resultado.shape == (4, 2)

    def test_n_components(self):
        pca = PCA()
        X = np.array([
            [1.5, 2.0, 0.3],
            [1.6, 1.9, 0.8],
            [8.5, 9.0, 4.1],
            [8.6, 9.1, 3.7],
            [4.2, 5.3, 2.2]
        ])

        pca.set_params(n_components=5)

        with pytest.raises(ValueError):
            pca.fit(X)

    def test_params(self):
        pca = PCA()
        pca.set_params(n_components=2)

        di = pca.get_params()

        assert type(di) == dict

        assert di['n_components'] == 2

    def test_attributes(self):
        pca = PCA()
        X_train = np.array([
            [1.5, 2.0, 0.3],
            [1.6, 1.9, 0.8],
            [8.5, 9.0, 4.1],
            [8.6, 9.1, 3.7],
            [4.2, 5.3, 2.2]
        ])

        n_samples = X_train.shape[0]
        n_features = X_train.shape[1]
        k = 2

        pca.set_params(n_components=k)
        pca.fit(X_train)

        assert isinstance(pca.components, np.ndarray)
        assert pca.components.shape == (k, n_features)

        assert isinstance(pca.explained_variance, np.ndarray)
        assert pca.explained_variance.shape == (k,)

        assert isinstance(pca.explained_variance_ratio, np.ndarray)
        assert pca.explained_variance_ratio.shape == (k,)

        assert isinstance(pca.mean, np.ndarray)
        assert pca.mean.shape == (n_features,)

        projetado = pca.transform(X_train)
        assert projetado.shape == (n_samples, k)

        reconstruido = pca.inverse_transform(projetado)
        assert isinstance(reconstruido, np.ndarray)
        assert reconstruido.shape == (n_samples, n_features)


class TestRightResults():
    X = np.array([
        [2.5, 2.4, 1.2],
        [0.5, 0.7, 0.3],
        [2.2, 2.9, 1.1],
        [1.9, 2.2, 0.8],
        [3.1, 3.0, 1.5],
        [2.3, 2.7, 1.0],
        [2.0, 1.6, 0.9],
        [1.0, 1.1, 0.4]
    ])

    def test_fit(self):
        pca = PCA()
        pca.set_params(n_components=2)
        pca.fit(self.X)

        np.testing.assert_allclose(
            pca.mean,
            np.array([1.9375, 2.075, 0.9]),
            atol=1e-9,
            err_msg="The mean do not match"
        )

        ExpectedComponents = np.array([
            [0.6625976315484233, 0.6787039132053335, 0.31674181420549335],
            [-0.6140670309155739, 0.734406065559754, -0.2890837463635315]
        ])

        np.testing.assert_allclose(
            pca.components,
            ExpectedComponents,
            atol=1e-9,
            err_msg="The components do not match"
        )

        np.testing.assert_allclose(
            pca.explained_variance,
            np.array([1.5116746234066816, 0.06373935157251642]),
            atol=1e-9,
            err_msg="The explained variance do not match"
        )

    def test_explained_variance_ratio(self):
        pca = PCA()
        pca.set_params(n_components=3)
        pca.fit(self.X)

        np.testing.assert_allclose(
            pca.explained_variance_ratio,
            np.array([0.957297058812328, 0.04036417152618929, 0.00233876966148274]),
            atol=1e-9,
            err_msg="The explained variance ratio do not match"
        )

        np.testing.assert_allclose(
            pca.explained_variance_ratio.sum(),
            1.0,
            atol=1e-9,
            err_msg="The explained variance ratio do not sum to 1"
        )

    def test_transform(self):
        pca = PCA()
        pca.set_params(n_components=2)
        pca.fit(self.X)

        ExpectedTrain = np.array([
            [0.6883124837993695, -0.19345585749214955],
            [-2.075747064531488, 0.04636326461459492],
            [0.7972109695169601, 0.3868756591987526],
            [0.02831639654705169, 0.14373664649065682],
            [1.588115954913272, -0.20797756061470096],
            [0.6960557686101865, 0.207496117631598],
            [-0.2809720068007568, -0.3872220705731062],
            [-1.4412925020545935, 0.00418380074435643]
        ])

        np.testing.assert_allclose(
            pca.transform(self.X),
            ExpectedTrain,
            atol=1e-9,
            err_msg="The transform do not match"
        )

        Data = np.array([[1.5, 1.0, 0.5], [3.0, 2.5, 1.4]])
        ExpectedData = np.array([
            [-1.1461898961803658, -0.405198695905759],
            [1.1508300537352134, -0.48486551566666736]
        ])

        np.testing.assert_allclose(
            pca.transform(Data),
            ExpectedData,
            atol=1e-9,
            err_msg="The transform do not match"
        )

    def test_inverse_transform(self):
        pca = PCA()
        pca.set_params(n_components=2)
        pca.fit(self.X)

        ExpectedReconstruction = np.array([
            [2.5123690855541057, 2.4000852211024166, 1.1739422888987017],
            [0.5336447591024065, 0.7002318072301053, 0.22912124272013845],
            [2.2281625128686766, 2.9001940353943705, 1.040670583851286],
            [1.8679984415417834, 2.1997795141703143, 0.8674170585568258],
            [3.11749403349353, 3.0001205312077044, 1.4631456611099656],
            [2.271288378826183, 2.6998021813336255, 1.0604862118969594],
            [1.9891089209442192, 1.5999249621356741, 0.9229440236608876],
            [0.979933867669096, 1.09986174742579, 0.4422729293052363]
        ])

        np.testing.assert_allclose(
            pca.inverse_transform(pca.transform(self.X)),
            ExpectedReconstruction,
            atol=1e-9,
            err_msg="The inverse transform do not match"
        )

        pca.set_params(n_components=3)
        pca.fit(self.X)

        np.testing.assert_allclose(
            pca.inverse_transform(pca.transform(self.X)),
            self.X,
            atol=1e-9,
            err_msg="The inverse transform do not match"
        )
