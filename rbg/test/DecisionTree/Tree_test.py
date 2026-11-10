from rbg.DecisionTree import ClassificationTree as Classification
from rbg.DecisionTree import RegressionTree as Regression
import numpy as np
import pandas as pd
import pytest

'''
# para impurity: Gini, classification (por padrão), entropy
features = 100
#testa se consegue receber esses parametros
Classification.set_params(impurity = 'Gini', min_inpurity_decrease = 0.1 , max_depth = 30)
#verifica se retorna um dicionario com as coisas corretas
Classification.get_params()

#verifica se da certo e faz os testes de tamanho de array etc
Classification.fit(X,Y)
#verifica se retorna o tamanho correto etc
Classification.predict(Data)


#regressão, igualmente
#testes iguais
features = 100
Regression.set_params(min_inpurity_decrease = 0.1 , max_depth = 30)
Regression.get_params()

Regression.fit(X,Y)
Regression.predict(Data)
'''

class TestClassificationTree:
    
    def test_params(self):
        modelo_classificacao = Classification()
        modelo_classificacao.set_params(criterion_split='Gini', min_impurity_decrease=0.1, max_depth=30)
        
        params = modelo_classificacao.get_params()
        
        assert isinstance(params, dict)
        assert params.get('criterion_split') == 'Gini'
        assert params.get('min_impurity_decrease') == 0.1
        assert params.get('max_depth') == 30

    def test_fit_and_predict(self):
        X = np.array([
            [1.5, 2.0], 
            [1.6, 1.9], 
            [8.5, 9.0], 
            [8.6, 9.1]
        ])
        Y = np.array(['classeA', 'classeA', 'classeB', 'classeB'])
        
        X_errado = np.zeros((10, 2))
        Y_errado = np.zeros(5)
        
        modelo_classificacao = Classification()
        
        with pytest.raises(ValueError, match="The data do not match"):
            modelo_classificacao.fit(X_errado, Y_errado)

        modelo_classificacao.fit(X, Y)

        Data = np.array([[1.5, 2.0], [8.5, 9.0]])
        resultado = modelo_classificacao.predict(Data)

        assert isinstance(resultado, np.ndarray)
        assert resultado.shape == (2,)
    def test_iniciate_class(self):

        with pytest.raises(ValueError, match="The only split criterions available for classification are Gini,classif_error,entropy, and Other"):
                  modelo_classificacao = Classification( max_depth = 34, max_leaf = 90, min_samples_leaf = 1, random_forest = False, min_impurity_decrease = 0.0001, criterion_split = "MAE",  min_samples_split = 2, custom_split = None, criterion_choose = "Standard", custom_choose = None,  criterion_evaluation = "Standard", custom_evaluation = None )

        modelo_classificacao = Classification(min_samples_split = 6, max_depth = 34, max_leaf = 90, min_samples_leaf = 1, random_forest = False, min_impurity_decrease = 0.0001, criterion_split = "entropy", custom_split = None, criterion_choose = "Standard", custom_choose = None,  criterion_evaluation = "Standard", custom_evaluation = None )
        params = modelo_classificacao.get_params()
        assert params.get('criterion_split') == 'entropy'
        assert params.get('min_impurity_decrease') == 0.0001
        assert params.get('max_depth') == 34
        assert params.get('min_samples_split') == 6
        assert params.get('max_leaf') == 90
        assert params.get('min_samples_leaf') == 1
        assert params.get('random_forest') == False
        assert params.get('max_depth') == 34
        assert params.get('criterion_choose') == "Standard"
        assert params.get('criterion_evaluation') == "Standard"

    def test_fit_class(self):
        modelo_classificacao = Classification()
        X = np.array([[[4],[5],[8]],[[6],[9],[18]]])
        Y = np.array([1,2])

        with pytest.raises(ValueError, match="X must be two-dimensional"):
            modelo_classificacao.fit(X, Y)
        X = np.array([[4, 5,8],[6,9,18]])
        Y = np.array([[4,9],[6,0]])
        with pytest.raises(ValueError, match="y must be one-dimensional."):
             modelo_classificacao.fit(X, Y)
        X = np.array([]).reshape(0, 2)
        Y = np.array([4,9])
        
        with pytest.raises(ValueError, match="X must contain at least one observation and one feature."):
            modelo_classificacao.fit(X, Y)
        X = np.array([]).reshape(2, 0)
        Y = np.array([4,9])
        
        with pytest.raises(ValueError, match="X must contain at least one observation and one feature."):
            modelo_classificacao.fit(X, Y)


        X = np.array([[6,9,0],[3,4,6]])
        Y = np.array([4])
        
        with pytest.raises(ValueError, match="The data do not match"):
            modelo_classificacao.fit(X, Y)
    def test_tree_args(self):
        modelo_classificacao = Classification()
        X = np.array([[1],[2],[10],[11]])
        Y = np.array([0,0,1,1])
        modelo_classificacao.fit(X, Y)

        dic = modelo_classificacao.get_tree_structure_info()
        assert isinstance(dic, dict)
        assert dic.get('n') == 7
        assert dic.get('children_left')[0] == 2
        assert dic.get('children_right')[0] == 1
        assert dic.get('children_left')[1] == -1
        assert dic.get('children_right')[2] == -1
        assert int(dic.get('impurity')[2]) == 0

    def test_predict(self):
        modelo_classificacao = Classification()
        X = np.array([[1],[2],[10],[11]])
        Y = np.array([0,0,1,1])
        modelo_classificacao.fit(X, Y)

        x_test = np.array([[3]])
        a = modelo_classificacao.predict(x_test)
        assert a == np.array([0])
        x_test = np.array([[3],[10]])
        a = modelo_classificacao.predict(x_test)
        assert np.array_equal(a, np.array([0, 1]))
    def test_prune(self):
        '''
        Creating data that an overfit would cause a lot of trouble, forcing the prune 
        method to choose to cut the tree (that is, to choose an alfa different from 0)
        '''
        np.random.seed(42)
        X_train = np.random.uniform(0, 10, size=(120, 2))
    
        y_train = ((np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])) > 0).astype(int)
    
        noise_idx = np.random.choice(120, size=30, replace=False)
        y_train[noise_idx] = 1 - y_train[noise_idx]  # creating an extreme bad overfitting 

        modelo_classificacao = Classification()
        modelo_classificacao.fit(X_train, y_train, prune=True)
    
        X_test = np.random.uniform(0, 10, size=(40, 2))
        Y_test_prune = modelo_classificacao.predict(X_test)

        modelo_classificacao_2 = Classification()
        modelo_classificacao_2.fit(X_train, y_train, prune=False)
        Y_test_no_prune = modelo_classificacao_2.predict(X_test)
       
        assert not np.array_equal(Y_test_no_prune, Y_test_prune)

    def test_predictability(self):
        np.random.seed(42)
        X_train = np.random.uniform(0, 10, size=(120, 2))
    
        y_train = ((np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])) > 0).astype(int)
    
        noise_idx = np.random.choice(120, size=30, replace=False)
        y_train[noise_idx] = 1 - y_train[noise_idx]  # creating an extreme bad overfitting 

        X_test = np.random.uniform(0, 10, size=(40, 2))

        modelo_classificacao = Classification()
        modelo_classificacao.fit(X_train, y_train, prune=True)
    
        Y_test_1 = modelo_classificacao.predict(X_test)

        modelo_classificacao_2 = Classification()
        modelo_classificacao_2.fit(X_train, y_train, prune = True)
        Y_test_2 = modelo_classificacao_2.predict(X_test)

        assert np.array_equal(Y_test_1, Y_test_2)


        modelo_classificacao = Classification()
        modelo_classificacao.fit(X_train, y_train)
    
        Y_test_1 = modelo_classificacao.predict(X_test)

        modelo_classificacao_2 = Classification()
        modelo_classificacao_2.fit(X_train, y_train)
        Y_test_2 = modelo_classificacao_2.predict(X_test)
       
        assert np.array_equal(Y_test_1, Y_test_2)


class TestRegressionTree:
    
    def test_params(self):
        modelo_regressao = Regression(criterion_split = "MAE", criterion_choose = "median")
        modelo_regressao.set_params(min_impurity_decrease=0.1, max_depth=30)
        
        params = modelo_regressao.get_params()
        
        assert isinstance(params, dict)
        assert params.get('min_impurity_decrease') == 0.1
        assert params.get('max_depth') == 30

    def test_fit_and_predict(self):
        X = np.array([
            [1.5, 2.0], 
            [1.6, 1.9], 
            [8.5, 9.0], 
            [8.6, 9.1]
        ])
        Y = np.array([10.5, 12.0, 20.5, 22.0])
        
        X_errado = np.zeros((10, 2))
        Y_errado = np.zeros(5)
        
        modelo_regressao = Regression()
        
        with pytest.raises(ValueError, match="The data do not match"):
            modelo_regressao.fit(X_errado, Y_errado)

        modelo_regressao.fit(X, Y)

        Data = np.array([[1.5, 2.0], [8.5, 9.0], [5.0, 5.0]])
        resultado = modelo_regressao.predict(Data)

        assert isinstance(resultado, np.ndarray)
        assert resultado.shape == (3,)
        Data = np.array([[1.5, 2.0], [8.5, 9.0], [5.0, 5.0]])
        resultado = modelo_regressao.predict(Data)

        assert isinstance(resultado, np.ndarray)
        assert resultado.shape == (3,)
    def test_iniciate_class(self):

        with pytest.raises(ValueError, match="The only split criterions available for regression are MSE, RSS, RMSE, MAE, and Other"):
                  model_reg = Regression( max_depth = 34, max_leaf = 90, min_samples_leaf = 1, random_forest = False, min_impurity_decrease = 0.0001, criterion_split = "Gini",  min_samples_split = 2, custom_split = None, criterion_choose = "Standard", custom_choose = None,  criterion_evaluation = "Standard", custom_evaluation = None )

        model_reg = Regression(min_samples_split = 6, max_depth = 34, max_leaf = 90, min_samples_leaf = 1, random_forest = False, min_impurity_decrease = 0.0001, criterion_split = "MAE", custom_split = None, criterion_choose = "median", custom_choose = None,  criterion_evaluation = "Standard", custom_evaluation = None )
        params = model_reg.get_params()
        assert params.get('criterion_split') == 'MAE'
        assert params.get('min_impurity_decrease') == 0.0001
        assert params.get('max_depth') == 34
        assert params.get('min_samples_split') == 6
        assert params.get('max_leaf') == 90
        assert params.get('min_samples_leaf') == 1
        assert params.get('random_forest') == False
        assert params.get('max_depth') == 34
        assert params.get('criterion_choose') == "median"
        assert params.get('criterion_evaluation') == "Standard"

    def test_fit_class(self):
        model_reg = Regression()
        X = np.array([[[4],[5],[8]],[[6],[9],[18]]])
        Y = np.array([1,2])

        with pytest.raises(ValueError, match="X must be two-dimensional"):
            model_reg.fit(X, Y)
        X = np.array([[4, 5,8],[6,9,18]])
        Y = np.array([[4,9],[6,0]])
        with pytest.raises(ValueError, match="y must be one-dimensional."):
             model_reg.fit(X, Y)
        X = np.array([]).reshape(0, 2)
        Y = np.array([4,9])
        
        with pytest.raises(ValueError, match="X must contain at least one observation and one feature."):
            model_reg.fit(X, Y)
        X = np.array([]).reshape(2, 0)
        Y = np.array([4,9])
        
        with pytest.raises(ValueError, match="X must contain at least one observation and one feature."):
            model_reg.fit(X, Y)


        X = np.array([[6,9,0],[3,4,6]])
        Y = np.array([4])
        
        with pytest.raises(ValueError, match="The data do not match"):
            model_reg.fit(X, Y)
    def test_tree_args(self):
        model_reg = Regression()
        X = np.array([[1],[2],[10],[11]])
        Y = np.array([0,0,1,1])
        model_reg.fit(X, Y)

        dic = model_reg.get_tree_structure_info()
        assert isinstance(dic, dict)
        assert dic.get('n') == 7
        assert dic.get('children_left')[0] == 2
        assert dic.get('children_right')[0] == 1
        assert dic.get('children_left')[1] == -1
        assert dic.get('children_right')[2] == -1
        assert int(dic.get('impurity')[2]) == 0

    def test_predict(self):
        model_reg = Regression()
        X = np.array([[1],[2],[10],[11]])
        Y = np.array([0,2,50,60])
        model_reg.fit(X, Y)

        x_test = np.array([[3]])
        a = model_reg.predict(x_test)
        assert a == np.array([2])
        x_test = np.array([[3],[10.8]])
        a = model_reg.predict(x_test)
        assert np.array_equal(a, np.array([2, 60]))

    def test_prune(self):
        '''
        Creating data that an overfit would cause a lot of trouble, forcing the prune 
        method to choose to cut the tree (that is, to choose an alfa different from 0)
        '''
        np.random.seed(42)
        X_train = np.random.uniform(0, 10, size=(120, 2))
    
        y_train = (np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])) 
    
        noise = np.random.normal(0, 0.5, size=120)
        y_train = y_train + noise

        model_reg = Regression()
        model_reg.fit(X_train, y_train, prune=True)
    
        X_test = np.random.uniform(0, 10, size=(40, 2))
        Y_test_prune = model_reg.predict(X_test)

        model_reg_2 = Regression()
        model_reg_2.fit(X_train, y_train, prune=False)
        Y_test_no_prune = model_reg_2.predict(X_test)
       

        assert not np.array_equal(Y_test_no_prune, Y_test_prune)
    def test_predictability(self):
        np.random.seed(42)
        X_train = np.random.uniform(0, 10, size=(120, 2))
    
        y_train = (np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])) 
    
        noise = np.random.normal(0, 0.5, size=120)
        y_train = y_train + noise

        model_reg = Regression()
        model_reg.fit(X_train, y_train, prune=True)
        model_reg2 = Regression()
        model_reg2.fit(X_train, y_train, prune=True)
        X_test = np.random.uniform(0, 10, size=(40, 2))
        y_pred1 = model_reg.predict(X_test)
        y_pred2 = model_reg2.predict(X_test)

        assert np.array_equal(y_pred1, y_pred2)

        model_reg = Regression()
        model_reg.fit(X_train, y_train)
        model_reg2 = Regression()
        model_reg2.fit(X_train, y_train)
        X_test = np.random.uniform(0, 10, size=(40, 2))
        y_pred1 = model_reg.predict(X_test)
        y_pred2 = model_reg2.predict(X_test)
        assert np.array_equal(y_pred1, y_pred2)



    
