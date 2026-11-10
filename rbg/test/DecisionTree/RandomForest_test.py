from rbg.DecisionTree import RandomForest as RdForest
from rbg.DecisionTree import RandomForestRegression as RdForestReg
from rbg.DecisionTree import Bagging
from rbg.DecisionTree import BaggingReg
import numpy as np
import pandas as pd
import pytest 

from rbg.Utilities.crossvalidation import train_val_test_split


'''
# para impurity: Gini, classification (por padrão), entropy
# m representa o tamanho das divisões de features =============== talvez de pra pegar automaticamente, ou os dois 
features = 100
RdForest.set_params(impurity = 'Gini', m = features/3,n_trees = 100, min_inpurity_decrease = 0.1 , max_depth = 30)
RdForest.get_params()

RdForest.fit(X,Y)
RdForest.predict(Data)


#regressão, igualmente
features = 100
RdForestReg.set_params(m = features/3, n_trees = 100, min_inpurity_decrease = 0.1 , max_depth = 30)
RdForestReg.get_params()

RdForestReg.fit(X,Y)
RdForestReg.predict(Data)
'''
def divi_3(m):
    return m/3

class TestRdForest_DataEntry:
    def test_DataFrame(self):
        X = pd.DataFrame({"col1": [1.0, 2.0, 3.0], "col2": [2.0, 3.0, 4.0]})
        Y = pd.Series(['classA', 'classA', 'classB']) 

        Data = pd.DataFrame({"col1": [1.5, 2.0, 0.5], "col2": [2.5, 3.0, 1.5]})

        modelo_rdforest = RdForest()
        modelo_rdforest.fit(X, Y)
        DtFrameResult = modelo_rdforest.predict(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)
        assert DtFrameResult.shape == (3,)

    def test_Numpy(self):
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        Y = np.array(['classA', 'classA', 'classB'])

        Data = np.array([[1.5, 2.5], [2.0, 3.0], [0.5, 1.5]])

        modelo_rdforest = RdForest()
        modelo_rdforest.fit(X, Y)
        NpResult = modelo_rdforest.predict(Data)

        assert isinstance(NpResult, np.ndarray)
        assert NpResult.shape == (3,)


class TestRdForest_Error:
    def test_MatchingDim(self):
        X = np.zeros((10, 3)) 
        Y = np.zeros(5)       
        
        modelo_rdforest = RdForest()

        with pytest.raises(ValueError, match="The data do not match"):
            modelo_rdforest.fit(X, Y)

        X_cut = X[:5, :] 
        modelo_rdforest.fit(X_cut, Y)

        Data = np.zeros((4, 3))
        result = modelo_rdforest.predict(Data)
        
        # Testa se retorna uma array numpy de tamanho compativel
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_params(self):
        features = 100
        modelo_rdforest = RdForest()
        modelo_rdforest.set_params(criterion_split='Gini', factor_selection = divi_3, n_trees=100, min_impurity_decrease=0.1, max_depth=30)
        
        params = modelo_rdforest.get_params()
        
        assert isinstance(params, dict)
        assert params.get('criterion_split') == 'Gini'
        assert params.get('factor_selection') == divi_3
        assert params.get('n_trees') == 100
        assert params.get('min_impurity_decrease') == 0.1
        assert params.get('max_depth') == 30

    def test_iniciate_class(self):
    
        with pytest.raises(ValueError, match="The only split criterions available for classification are Gini,classif_error,entropy, and Other"):
            model_classificationRand = RdForest( max_depth = 34, max_leaf = 90, min_samples_leaf = 1, random_forest = False, n_trees = 900000, min_impurity_decrease = 0.0001, criterion_split = "MAE",  min_samples_split = 2, custom_split = None, criterion_choose = "Standard", custom_choose = None,  criterion_evaluation = "Standard", custom_evaluation = None )
        with pytest.raises(TypeError, match="number of trees must be integer"):
            model_classificationRand = RdForest(n_trees = 4.7)
        with pytest.raises(ValueError, match="number of trees must be at least one"):
            model_classificationRand = RdForest(n_trees = 0)
    
        model_classificationRand = RdForest(min_samples_split = 6, max_depth = 34, max_leaf = 90, min_samples_leaf = 1,n_trees = 900000, random_forest = True, min_impurity_decrease = 0.0001, criterion_split = "entropy", custom_split = None, criterion_choose = "Standard", custom_choose = None,  criterion_evaluation = "Standard", custom_evaluation = None )
        params = model_classificationRand.get_params()
        assert params.get('criterion_split') == 'entropy'
        assert params.get('min_impurity_decrease') == 0.0001
        assert params.get('max_depth') == 34
        assert params.get('min_samples_split') == 6
        assert params.get('max_leaf') == 90
        assert params.get('min_samples_leaf') == 1
        assert params.get('random_forest') == True
        assert params.get('max_depth') == 34
        assert params.get('criterion_choose') == "Standard"
        assert params.get('criterion_evaluation') == "Standard"
        assert params.get('n_trees') == 900000
    
    def test_fit_class(self):
        model_classificationRand = RdForest(n_trees = 20)
        X = np.array([[[4],[5],[8]],[[6],[9],[18]]])
        Y = np.array([1,2])

    
        with pytest.raises(ValueError, match="X must be two-dimensional"):
            model_classificationRand.fit(X, Y)
        X = np.array([[4, 5,8],[6,9,18]])
        Y = np.array([[4,9],[6,0]])
        with pytest.raises(ValueError, match="y must be one-dimensional."):
            model_classificationRand.fit(X, Y)
        X = np.array([]).reshape(0, 2)
        Y = np.array([4,9])
            
        with pytest.raises(ValueError, match="X must contain at least one observation and one feature."):
            model_classificationRand.fit(X, Y)
        X = np.array([]).reshape(2, 0)
        Y = np.array([4,9])
            
        with pytest.raises(ValueError, match="X must contain at least one observation and one feature."):
            model_classificationRand.fit(X, Y)
    
    
        X = np.array([[6,9,0],[3,4,6]])
        Y = np.array([4])
            
        with pytest.raises(ValueError, match="The data do not match"):
            model_classificationRand.fit(X, Y)
    def test_tree_args(self):
        model_classificationRand = RdForest(n_trees = 9)
        X = np.array([[1],[2],[10],[11]])
        Y = np.array([0,0,1,1])
        model_classificationRand.fit(X, Y)
    
        dic = model_classificationRand.get_tree_structure_info()# should return empty because it will not be only
        #one tree. There is no "original tree"
        assert isinstance(dic, dict)
        assert dic.get('n') == None
        assert dic.get('children_left') == None
        assert dic.get('children_right')== None

    
    def test_Random_vs_Bagging(self):
 
        np.random.seed(42)


        X = np.random.uniform(-2.0, 2.0, size=(200, 9))
        
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        X_train, X_val, X_test, y_train, y_val, y_test= train_val_test_split(X, y , val_size = 0, test_size = 0.2, shuffle = True, random_state = 42)

    
        model_bagging = Bagging(n_trees = 15)
        model_RandomFalse = RdForest(random_forest=False, n_trees = 15)
        model_Random = RdForest(n_trees = 15)
        model_bagging.fit(X_train, y_train)
        model_RandomFalse.fit(X_train, y_train)
        model_Random.fit(X_train,y_train)


        Y_class_RandomFalse = model_RandomFalse.predict(X_test)
        Y_class_bagging = model_bagging.predict(X_test)
        Y_class_Random = model_Random.predict(X_test)
           
    
        assert not np.array_equal(Y_class_bagging, Y_class_Random)
        assert np.array_equal(Y_class_RandomFalse, Y_class_bagging)
    
    
    def test_predictability(self):
        np.random.seed(42)


        X_train = np.random.uniform(0, 10, size=(120, 8))


        decision_boundary = (
            np.sin(X_train[:, 0]) + 
            np.cos(X_train[:, 1]) + 
            0.5 * X_train[:, 2] - 
            0.5 * X_train[:, 3]
        )
        y_train = (decision_boundary > 0).astype(int)


        noise_idx = np.random.choice(120, size=30, replace=False)
        y_train[noise_idx] = 1 - y_train[noise_idx]  


        X_test = np.random.uniform(0, 10, size=(40, 8))

    
    
        model_classificationRand = RdForest(n_trees = 30)
        model_classificationRand.fit(X_train, y_train)
        
        Y_test_1 = model_classificationRand.predict(X_test)
    
        model_classificationRand_2 = RdForest(n_trees = 30)
        model_classificationRand_2.fit(X_train, y_train)
        Y_test_2 = model_classificationRand_2.predict(X_test)
           
        assert np.array_equal(Y_test_1, Y_test_2)
    def test_OOB(self):
        np.random.seed(42)


        X_train = np.random.uniform(0, 10, size=(120, 8))


        decision_boundary = (
            np.sin(X_train[:, 0]) + 
            np.cos(X_train[:, 1]) + 
            0.5 * X_train[:, 2] - 
            0.5 * X_train[:, 3]
        )
        y_train = (decision_boundary > 0).astype(int)


        noise_idx = np.random.choice(120, size=30, replace=False)
        y_train[noise_idx] = 1 - y_train[noise_idx]  


    
    
    
        model_classificationRand = RdForest(n_trees = 50)
        model_classificationRand.fit(X_train, y_train)

        error = model_classificationRand.OOB()

        assert isinstance(error, (float, int, np.integer, np.floating))

        model_classificationRandBag = Bagging(n_trees = 50)
        model_classificationRandBag.fit(X_train, y_train)

        errorBag = model_classificationRandBag.OOB()

        assert error<= errorBag


#teste regressão
class TestRdForestReg_DataEntry:
    def test_DataFrame(self):
        X = pd.DataFrame({"col1": [1.0, 2.0, 3.0], "col2": [2.0, 3.0, 4.0]})
        Y = pd.Series([10.5, 20.5, 30.5]) 

        Data = pd.DataFrame({"col1": [1.5, 2.0, 0.5], "col2": [2.5, 3.0, 1.5]})

        modelo_rdforest_reg = RdForestReg()
        modelo_rdforest_reg.fit(X, Y)
        DtFrameResult = modelo_rdforest_reg.predict(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)
        assert DtFrameResult.shape == (3,)

    def test_Numpy(self):
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        Y = np.array([10.5, 20.5, 30.5])

        Data = np.array([[1.5, 2.5], [2.0, 3.0], [0.5, 1.5]])

        modelo_rdforest_reg = RdForestReg()
        modelo_rdforest_reg.fit(X, Y)
        NpResult = modelo_rdforest_reg.predict(Data)

        assert isinstance(NpResult, np.ndarray)
        assert NpResult.shape == (3,)


class TestRdForestReg_Error:
    def test_MatchingDim(self):
        X = np.zeros((10, 3)) 
        Y = np.zeros(5)       
        
        modelo_rdforest_reg = RdForestReg()

        with pytest.raises(ValueError, match="The data do not match"):
            modelo_rdforest_reg.fit(X, Y)

        X_cut = X[:5, :] 
        modelo_rdforest_reg.fit(X_cut, Y)

        Data = np.zeros((4, 3))
        result = modelo_rdforest_reg.predict(Data)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_params(self):
        features = 100
        modelo_rdforest_reg = RdForestReg()
        modelo_rdforest_reg.set_params( n_trees=100, min_impurity_decrease=0.1, max_depth=30)
        
        params = modelo_rdforest_reg.get_params()

        
        
        assert isinstance(params, dict)
        assert params.get('factor_selection') == np.sqrt
        assert params.get('n_trees') == 100
        assert params.get('min_impurity_decrease') == 0.1
        assert params.get('max_depth') == 30
    def test_iniciate_class(self):
    
        with pytest.raises(ValueError, match="The only split criterions available for regression are MSE, RSS, RMSE, MAE, and Other"):
            model_regressionRand = RdForestReg( max_depth = 34, max_leaf = 90, min_samples_leaf = 1, random_forest = True, n_trees = 900000, min_impurity_decrease = 0.0001, criterion_split = "Gini",  min_samples_split = 2, custom_split = None, criterion_choose = "Standard", custom_choose = None,  criterion_evaluation = "Standard", custom_evaluation = None )
        with pytest.raises(TypeError, match="number of trees must be integer"):
            model_regressionRand = RdForestReg(n_trees = 4.7)
        with pytest.raises(ValueError, match="number of trees must be at least one"):
            model_regressionRand = RdForestReg(n_trees = 0)
    
        model_regressionRand = RdForestReg(min_samples_split = 6, max_depth = 34, max_leaf = 90, min_samples_leaf = 1,n_trees = 900000, random_forest = True, min_impurity_decrease = 0.0001, criterion_split = "MAE", custom_split = None, criterion_choose = "Standard", custom_choose = None,  criterion_evaluation = "Standard", custom_evaluation = None )
        params = model_regressionRand.get_params()
        assert params.get('criterion_split') == 'MAE'
        assert params.get('min_impurity_decrease') == 0.0001
        assert params.get('max_depth') == 34
        assert params.get('min_samples_split') == 6
        assert params.get('max_leaf') == 90
        assert params.get('min_samples_leaf') == 1
        assert params.get('random_forest') == True
        assert params.get('max_depth') == 34
        assert params.get('criterion_choose') == "Standard"
        assert params.get('criterion_evaluation') == "Standard"
        assert params.get('n_trees') == 900000
    
    def test_fit_class(self):
        model_regressionRand = RdForestReg(n_trees = 9)
        X = np.array([[[4],[5],[8]],[[6],[9],[18]]])
        Y = np.array([1,2])

    
        with pytest.raises(ValueError, match="X must be two-dimensional"):
            model_regressionRand.fit(X, Y)
        X = np.array([[4, 5,8],[6,9,18]])
        Y = np.array([[4,9],[6,0]])
        with pytest.raises(ValueError, match="y must be one-dimensional."):
            model_regressionRand.fit(X, Y)
        X = np.array([]).reshape(0, 2)
        Y = np.array([4,9])
            
        with pytest.raises(ValueError, match="X must contain at least one observation and one feature."):
            model_regressionRand.fit(X, Y)
        X = np.array([]).reshape(2, 0)
        Y = np.array([4,9])
            
        with pytest.raises(ValueError, match="X must contain at least one observation and one feature."):
            model_regressionRand.fit(X, Y)
    
    
        X = np.array([[6,9,0],[3,4,6]])
        Y = np.array([4])
            
        with pytest.raises(ValueError, match="The data do not match"):
            model_regressionRand.fit(X, Y)
    def test_tree_args(self):
        model_regressionRand = RdForestReg(n_trees = 3)
        X = np.array([[1],[2],[10],[11]])
        Y = np.array([0,0,1,1])
        model_regressionRand.fit(X, Y)
    
        dic = model_regressionRand.get_tree_structure_info()# should return empty because it will not be only
        #one tree. There is no "original tree"
        assert isinstance(dic, dict)
        assert dic.get('n') == None
        assert dic.get('children_left') == None
        assert dic.get('children_right')== None
    
    
    def test_Bagging_vs_Random(self):

        np.random.seed(42)

        X_train = np.random.uniform(0, 10, size=(200, 15))
        y_train = 3.5 * X_train[:, 0] - 2.0 * X_train[:, 5] + 1.2 * X_train[:, 10] + np.random.normal(0, 1, size=200)
        X_test = np.random.uniform(0, 10, size=(100, 15))
        

    
        model_regressionRand = RdForestReg(random_forest= False)
        model_regressionRand2 = RdForestReg()
        model_baggin_classification = BaggingReg()
        model_regressionRand.fit(X_train, y_train)
        model_regressionRand2.fit(X_train, y_train)
        model_baggin_classification.fit(X_train,y_train)


        Y_class_tree2 = model_regressionRand2.predict(X_test)
        Y_class_tree1 = model_regressionRand.predict(X_test)
        Y_class_bagging = model_baggin_classification.predict(X_test)
           
    
        assert np.array_equal(Y_class_tree1, Y_class_bagging)
        assert not np.array_equal(Y_class_tree2, Y_class_bagging)
    
    
    def test_predictability(self):
        np.random.seed(42)
        X_train = np.random.uniform(0, 10, size=(200, 15))
        y_train = 3.5 * X_train[:, 0] - 2.0 * X_train[:, 5] + 1.2 * X_train[:, 10] + np.random.normal(0, 1, size=200)
        X_test = np.random.uniform(0, 10, size=(100, 15))
    
    
        model_regressionRand = RdForestReg(n_trees = 50)
        model_regressionRand.fit(X_train, y_train)
        
        Y_test_1 = model_regressionRand.predict(X_test)
    
        model_regressionRand_2 = RdForestReg(n_trees = 50)
        model_regressionRand_2.fit(X_train, y_train)
        Y_test_2 = model_regressionRand_2.predict(X_test)
           
        assert np.array_equal(Y_test_1, Y_test_2)
    def test_OOB(self):
        np.random.seed(42)
        X_train = np.random.uniform(0, 10, size=(200, 15))
        y_train = 3.5 * X_train[:, 0] - 2.0 * X_train[:, 5] + 1.2 * X_train[:, 10] + np.random.normal(0, 1, size=200)
    
    
        model_regressionRand = RdForestReg(n_trees = 50)
        model_regressionRand.fit(X_train, y_train)

        error = model_regressionRand.OOB()

        assert isinstance(error, (float, int, np.integer, np.floating))
        
        model_BagReg = BaggingReg(n_trees = 50)
        model_BagReg.fit(X_train, y_train)

        errorBag = model_BagReg.OOB()

        assert error<= errorBag
