from rbg.DecisionTree import Bagging
from rbg.DecisionTree import BaggingReg
import numpy as np
import pandas as pd
import pytest
from rbg.DecisionTree import ClassificationTree
from rbg.DecisionTree import RegressionTree 


from rbg.Utilities.crossvalidation import train_val_test_split


'''
# para impurity: Gini, classification (por padrão), entropy
features = 100
Bagging.set_params(impurity = 'Gini', n_trees = 100, min_inpurity_decrease = 0.1 , max_depth = 30)
Bagging.get_params()

Bagging.fit(X,Y)
Bagging.predict(Data)


#regressão, igualmente
features = 100
BaggingReg.set_params(n_trees = 100, min_inpurity_decrease = 0.1 , max_depth = 30)
BaggingReg.get_params()

BaggingReg.fit(X,Y)
BaggingReg.predict(Data)
'''


class TestBagging_DataEntry:
    def test_DataFrame(self):
        X = pd.DataFrame({"col1": [1.0, 2.0, 3.0], "col2": [2.0, 3.0, 4.0]})
        Y = pd.Series(['classA', 'classA', 'classB']) 

        Data = pd.DataFrame({"col1": [1.5, 2.0, 0.5], "col2": [2.5, 3.0, 1.5]})

        modelo_bagging = Bagging()
        modelo_bagging.fit(X, Y)
        DtFrameResult = modelo_bagging.predict(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)
        assert DtFrameResult.shape == (3,)

    def test_Numpy(self):
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        Y = np.array(['classA', 'classA', 'classB'])

        Data = np.array([[1.5, 2.5], [2.0, 3.0], [0.5, 1.5]])

        modelo_bagging = Bagging()
        modelo_bagging.fit(X, Y)
        NpResult = modelo_bagging.predict(Data)

        assert isinstance(NpResult, np.ndarray)
        assert NpResult.shape == (3,)


class TestBagging_Error:
    def test_MatchingDim(self):
        X = np.zeros((10, 3)) 
        Y = np.zeros(5)       
        
        modelo_bagging = Bagging(criterion_split='classif_error')

        with pytest.raises(ValueError, match="The data do not match"):
            modelo_bagging.fit(X, Y)

        X_cut = X[:5, :] 
        modelo_bagging.fit(X_cut, Y)

        Data = np.zeros((4, 3))
        result = modelo_bagging.predict(Data)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_params(self):
        modelo_bagging = Bagging()
        modelo_bagging.set_params(criterion_split='classif_error', n_trees=10, min_impurity_decrease=0.1, max_depth=30)
        
        params = modelo_bagging.get_params()
        
        assert isinstance(params, dict)
        assert params.get('criterion_split') == 'classif_error'
        assert params.get('n_trees') == 10
        assert params.get('min_impurity_decrease') == 0.1
        assert params.get('max_depth') == 30

    def test_iniciate_class(self):
    
        with pytest.raises(ValueError, match="The only split criterions available for classification are Gini,classif_error,entropy, and Other"):
            model_classificationBag = Bagging( max_depth = 34, max_leaf = 90, min_samples_leaf = 1, random_forest = False, n_trees = 900000, min_impurity_decrease = 0.0001, criterion_split = "MAE",  min_samples_split = 2, custom_split = None, criterion_choose = "Standard", custom_choose = None,  criterion_evaluation = "Standard", custom_evaluation = None )
        with pytest.raises(TypeError, match="number of trees must be integer"):
            model_classificationBag = Bagging(n_trees = 4.7)
        with pytest.raises(ValueError, match="number of trees must be at least one"):
            model_classificationBag= Bagging(n_trees = 0)
    
        model_classificationBag= Bagging(min_samples_split = 6, max_depth = 34, max_leaf = 90, min_samples_leaf = 1,n_trees = 900000, random_forest = False, min_impurity_decrease = 0.0001, criterion_split = "entropy", custom_split = None, criterion_choose = "Standard", custom_choose = None,  criterion_evaluation = "Standard", custom_evaluation = None )
        params = model_classificationBag.get_params()
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
        assert params.get('n_trees') == 900000
    
    def test_fit_class(self):
        model_classificationBag= Bagging(n_trees = 20)
        X = np.array([[[4],[5],[8]],[[6],[9],[18]]])
        Y = np.array([1,2])

    
        with pytest.raises(ValueError, match="X must be two-dimensional"):
            model_classificationBag.fit(X, Y)
        X = np.array([[4, 5,8],[6,9,18]])
        Y = np.array([[4,9],[6,0]])
        with pytest.raises(ValueError, match="y must be one-dimensional."):
            model_classificationBag.fit(X, Y)
        X = np.array([]).reshape(0, 2)
        Y = np.array([4,9])
            
        with pytest.raises(ValueError, match="X must contain at least one observation and one feature."):
            model_classificationBag.fit(X, Y)
        X = np.array([]).reshape(2, 0)
        Y = np.array([4,9])
            
        with pytest.raises(ValueError, match="X must contain at least one observation and one feature."):
            model_classificationBag.fit(X, Y)
    
    
        X = np.array([[6,9,0],[3,4,6]])
        Y = np.array([4])
            
        with pytest.raises(ValueError, match="The data do not match"):
            model_classificationBag.fit(X, Y)
    def test_tree_args(self):
        model_classificationBag= Bagging(n_trees = 9)
        X = np.array([[1],[2],[10],[11]])
        Y = np.array([0,0,1,1])
        model_classificationBag.fit(X, Y)
    
        dic = model_classificationBag.get_tree_structure_info()# should return empty because it will not be only
        #one tree. There is no "original tree"
        assert isinstance(dic, dict)
        assert dic.get('n') == None
        assert dic.get('children_left') == None
        assert dic.get('children_right')== None
    
    def test_predict(self):
        model_classificationBag= Bagging(n_trees = 9)
        X = np.array([[1],[2],[10],[11]])
        Y = np.array([0,0,1,1])
        model_classificationBag.fit(X, Y)
    
        x_test = np.array([[3]])
        a = model_classificationBag.predict(x_test)
        assert a == np.array([0])
        x_test = np.array([[3],[10]])
        a = model_classificationBag.predict(x_test)
        assert np.array_equal(a, np.array([0, 1]))
    
    def test_Bagging_vs_Tree(self):

        np.random.seed(42)
        centers = [
            np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
            np.array([8.0, 8.0, 8.0, 8.0, 8.0, 8.0]),
            np.array([2.0, 8.0, 2.0, 8.0, 2.0, 8.0]),
            np.array([8.0, 2.0, 8.0, 2.0, 8.0, 2.0])
        ]
        
        X_list = []
        y_list = []
        
        for class_label, center in enumerate(centers):
            X_class = np.random.normal(loc=center, scale=1.2, size=(150, 6))
            y_class = np.full(150, class_label, dtype=int)
            X_list.append(X_class)
            y_list.append(y_class)
            
        X = np.vstack(X_list)
        y = np.concatenate(y_list)

        X_train, X_val, X_test, y_train, y_val, y_test= train_val_test_split(X, y , val_size = 0, test_size = 0.2, shuffle = True, random_state = 42)

    
        model_classification= ClassificationTree()
        model_classification2 = ClassificationTree()
        model_baggin_classification = Bagging()
        model_classification.fit(X_train, y_train, prune=True)
        model_classification2.fit(X_train, y_train)
        model_baggin_classification.fit(X_train,y_train)


        Y_class_tree2 = model_classification2.predict(X_test)
        Y_class_tree1 = model_classification.predict(X_test)
        Y_class_bagging = model_baggin_classification.predict(X_test)
           
    
        assert not np.array_equal(Y_class_tree1, Y_class_bagging)
        assert not np.array_equal(Y_class_tree2, Y_class_bagging)
    
    
    def test_predictability(self):
        np.random.seed(42)
        X_train = np.random.uniform(0, 10, size=(120, 2))
        
        y_train = ((np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])) > 0).astype(int)
        
        noise_idx = np.random.choice(120, size=30, replace=False)
        y_train[noise_idx] = 1 - y_train[noise_idx]  # creating an extreme bad overfitting 
    
        X_test = np.random.uniform(0, 10, size=(40, 2))
    
    
        model_classificationBag = Bagging(n_trees = 50)
        model_classificationBag.fit(X_train, y_train)
        
        Y_test_1 = model_classificationBag.predict(X_test)
    
        model_classificationBag_2 = Bagging(n_trees = 50)
        model_classificationBag_2.fit(X_train, y_train)
        Y_test_2 = model_classificationBag_2.predict(X_test)
           
        assert np.array_equal(Y_test_1, Y_test_2)
    def test_OOB(self):
        np.random.seed(42)
        X_train = np.random.uniform(0, 10, size=(120, 2))
        
        y_train = ((np.sin(X_train[:, 0]) + np.cos(X_train[:, 1])) > 0).astype(int)
        
        noise_idx = np.random.choice(120, size=30, replace=False)
        y_train[noise_idx] = 1 - y_train[noise_idx]  # creating an extreme bad overfitting 
    
        X_test = np.random.uniform(0, 10, size=(40, 2))
    
    
        model_classificationBag = Bagging(n_trees = 50)
        model_classificationBag.fit(X_train, y_train)

        error = model_classificationBag.OOB()

        assert isinstance(error, (float, int, np.integer, np.floating))

class TestBaggingReg_DataEntry:
    def test_DataFrame(self):
        X = pd.DataFrame({"col1": [1.0, 2.0, 3.0], "col2": [2.0, 3.0, 4.0]})
        Y = pd.Series([10.5, 20.5, 30.5]) 

        Data = pd.DataFrame({"col1": [1.5, 2.0, 0.5], "col2": [2.5, 3.0, 1.5]})

        modelo_bagging_reg = BaggingReg()
        modelo_bagging_reg.fit(X, Y)
        DtFrameResult = modelo_bagging_reg.predict(Data)
        
        assert isinstance(DtFrameResult, np.ndarray)
        assert DtFrameResult.shape == (3,)

    def test_Numpy(self):
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        Y = np.array([10.5, 20.5, 30.5])

        Data = np.array([[1.5, 2.5], [2.0, 3.0], [0.5, 1.5]])

        modelo_bagging_reg = BaggingReg()
        modelo_bagging_reg.fit(X, Y)
        NpResult = modelo_bagging_reg.predict(Data)

        assert isinstance(NpResult, np.ndarray)
        assert NpResult.shape == (3,)


class TestBaggingReg_Error:
    def test_MatchingDim(self):
        X = np.zeros((10, 3)) 
        Y = np.zeros(5)       
        
        modelo_bagging_reg = BaggingReg()

        with pytest.raises(ValueError, match="The data do not match"):
            modelo_bagging_reg.fit(X, Y)

        X_cut = X[:5, :] 
        modelo_bagging_reg.fit(X_cut, Y)

        Data = np.zeros((4, 3))
        result = modelo_bagging_reg.predict(Data)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)

    def test_params(self):
        modelo_bagging_reg = BaggingReg(criterion_split = "MAE", criterion_choose = "median")
        modelo_bagging_reg.set_params(n_trees=101, min_impurity_decrease=0.1, max_depth=30)
        
        params = modelo_bagging_reg.get_params()
        
        assert isinstance(params, dict)
        assert params.get('n_trees') == 101
        assert params.get('min_impurity_decrease') == 0.1
        assert params.get('max_depth') == 30
        params = modelo_bagging_reg.get_params()
    def test_iniciate_class(self):
    
        with pytest.raises(ValueError, match="The only split criterions available for regression are MSE, RSS, RMSE, MAE, and Other"):
            model_regressionBag = BaggingReg( max_depth = 34, max_leaf = 90, min_samples_leaf = 1, random_forest = False, n_trees = 900000, min_impurity_decrease = 0.0001, criterion_split = "Gini",  min_samples_split = 2, custom_split = None, criterion_choose = "Standard", custom_choose = None,  criterion_evaluation = "Standard", custom_evaluation = None )
        with pytest.raises(TypeError, match="number of trees must be integer"):
            model_regressionBag = BaggingReg(n_trees = 4.7)
        with pytest.raises(ValueError, match="number of trees must be at least one"):
            model_regressionBag = BaggingReg(n_trees = 0)
    
        model_regressionBag = BaggingReg(min_samples_split = 6, max_depth = 34, max_leaf = 90, min_samples_leaf = 1,n_trees = 900000, random_forest = False, min_impurity_decrease = 0.0001, criterion_split = "MAE", custom_split = None, criterion_choose = "Standard", custom_choose = None,  criterion_evaluation = "Standard", custom_evaluation = None )
        params = model_regressionBag.get_params()
        assert params.get('criterion_split') == 'MAE'
        assert params.get('min_impurity_decrease') == 0.0001
        assert params.get('max_depth') == 34
        assert params.get('min_samples_split') == 6
        assert params.get('max_leaf') == 90
        assert params.get('min_samples_leaf') == 1
        assert params.get('random_forest') == False
        assert params.get('max_depth') == 34
        assert params.get('criterion_choose') == "Standard"
        assert params.get('criterion_evaluation') == "Standard"
        assert params.get('n_trees') == 900000
    
    def test_fit_class(self):
        model_regressionBag = BaggingReg(n_trees = 9)
        X = np.array([[[4],[5],[8]],[[6],[9],[18]]])
        Y = np.array([1,2])

    
        with pytest.raises(ValueError, match="X must be two-dimensional"):
            model_regressionBag.fit(X, Y)
        X = np.array([[4, 5,8],[6,9,18]])
        Y = np.array([[4,9],[6,0]])
        with pytest.raises(ValueError, match="y must be one-dimensional."):
            model_regressionBag.fit(X, Y)
        X = np.array([]).reshape(0, 2)
        Y = np.array([4,9])
            
        with pytest.raises(ValueError, match="X must contain at least one observation and one feature."):
            model_regressionBag.fit(X, Y)
        X = np.array([]).reshape(2, 0)
        Y = np.array([4,9])
            
        with pytest.raises(ValueError, match="X must contain at least one observation and one feature."):
            model_regressionBag.fit(X, Y)
    
    
        X = np.array([[6,9,0],[3,4,6]])
        Y = np.array([4])
            
        with pytest.raises(ValueError, match="The data do not match"):
            model_regressionBag.fit(X, Y)
    def test_tree_args(self):
        model_regressionBag = BaggingReg(n_trees = 3)
        X = np.array([[1],[2],[10],[11]])
        Y = np.array([0,0,1,1])
        model_regressionBag.fit(X, Y)
    
        dic = model_regressionBag.get_tree_structure_info()# should return empty because it will not be only
        #one tree. There is no "original tree"
        assert isinstance(dic, dict)
        assert dic.get('n') == None
        assert dic.get('children_left') == None
        assert dic.get('children_right')== None
    
    
    def test_Bagging_vs_Tree(self):

        np.random.seed(42)

        X_train = np.random.uniform(0, 10, size=(200, 15))
        y_train = 3.5 * X_train[:, 0] - 2.0 * X_train[:, 5] + 1.2 * X_train[:, 10] + np.random.normal(0, 1, size=200)
        X_test = np.random.uniform(0, 10, size=(100, 15))
        

    
        model_regression= RegressionTree()
        model_regression2 = RegressionTree()
        model_baggin_regression = BaggingReg()
        model_regression.fit(X_train, y_train, prune=True)
        model_regression2.fit(X_train, y_train)
        model_baggin_regression.fit(X_train,y_train)


        Y_class_tree2 = model_regression2.predict(X_test)
        Y_class_tree1 = model_regression.predict(X_test)
        Y_class_bagging = model_baggin_regression.predict(X_test)
           
    
        assert not np.array_equal(Y_class_tree1, Y_class_bagging)
        assert not np.array_equal(Y_class_tree2, Y_class_bagging)
    
    
    def test_predictability(self):
        np.random.seed(42)
        X_train = np.random.uniform(0, 10, size=(200, 15))
        y_train = 3.5 * X_train[:, 0] - 2.0 * X_train[:, 5] + 1.2 * X_train[:, 10] + np.random.normal(0, 1, size=200)
        X_test = np.random.uniform(0, 10, size=(100, 15))
    
    
        model_regressionBag = Bagging(n_trees = 50)
        model_regressionBag.fit(X_train, y_train)
        
        Y_test_1 = model_regressionBag.predict(X_test)
    
        model_regressionBag_2 = Bagging(n_trees = 50)
        model_regressionBag_2.fit(X_train, y_train)
        Y_test_2 = model_regressionBag_2.predict(X_test)
           
        assert np.array_equal(Y_test_1, Y_test_2)
    def test_OOB(self):
        np.random.seed(42)
        X_train = np.random.uniform(0, 10, size=(200, 15))
        y_train = 3.5 * X_train[:, 0] - 2.0 * X_train[:, 5] + 1.2 * X_train[:, 10] + np.random.normal(0, 1, size=200)
    
    
        model_regressionBag = Bagging(n_trees = 50)
        model_regressionBag.fit(X_train, y_train)

        error = model_regressionBag.OOB()

        assert isinstance(error, (float, int, np.integer, np.floating))

