import numpy as np
from rbg.Utilities import MSE, rss, RMSE, MAE
from rbg.Utilities import Gini as gini, entropy as ent, classif_error as c_e, mode
from rbg.ModelSelection import KFold
from rbg.Utilities import Resample as resample

from ._ConstructionTree import ConstructionTree

def Gini(_, y_true):
    return gini(y_true)
def entropy(_, y_true):
    return ent(y_true)
def classif_error(_, y_true):
    return c_e(y_true)


epsolon = 0.000000000001 
    

                         

class TreeBase(ConstructionTree):
    def __init__(self, min_impurity_decrease = 0.01, min_samples_split = 2, max_depth = None, min_samples_leaf = 1, max_leaf = None, random_forest = False, criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", custom_choose = None,  criterion_evaluation = "Standard", custom_evaluation = None, factor_selection = np.sqrt , random_seed = 0, form = "regression" ): 
        """
        Args:
        
            min_impurity_decrease: The minimum purity gain to allow a split in a node.
            min_samples_split: The minimum number of samples in a node to allow a split.
            max_depth: The maximum depth the tree can grow.
            min_samples_leaf: The minimum number of samples required for each leaf node.
            max_leaf: Maximum number of leaves of the tree. 
            random_forest: Boolean indicating whether it is a random forest or not. 
            criterion_split: String indicating the metric used to calculate the split.
            custom_split: Metric function to calculate the split.
            criterion_choose: String indicating the metric used to choose the predicted value.
            custom_choose: Metric function to choose the predicted value.
            criterion_evaluation: String indicating the metric used to evaluate the error.
            custom_evaluation: Metric function to evaluate the error.
            factor_selection: Function to select the amount of features that can be selected to split a node.
            random_seed: Random seed associated with the tree.
            form: Indicates whether it is a classification or regression tree.
        """
        super().__init__( min_impurity_decrease =min_impurity_decrease , min_samples_split = min_samples_split , max_depth = max_depth  , min_samples_leaf =min_samples_leaf , max_leaf =  max_leaf , random_forest = random_forest, criterion_split = criterion_split , custom_split = custom_split , criterion_choose = criterion_choose , custom_choose = custom_choose, criterion_evaluation = criterion_evaluation , custom_evaluation = custom_evaluation, factor_selection = factor_selection,random_seed =random_seed, form=form  )
    def fit(self,X,Y, prune =False):
        """
        Build the decision tree from the training data, with optional cost-complexity pruning.

        Args:
            X : Matrix of shape (n_samples, n_features).
            Y : Corresponding vector of shape (n_samples,).
            prune : If True, applies Minimal Cost-Complexity Pruning using K-Fold cross-validation. Defaults to False.
        """
        super()._tree_construction(X,Y)
        if prune:
            self._prune(X,Y)
    def _prune(self,X,Y):
        """

        Performs Minimal Cost-Complexity Pruning using K-Fold cross-validation.

        This method determines the optimal complexity penalty parameter (alpha) by:


        1 - It takes the list of possible "penalty parameter" values and its associated trees from
        the original main tree calculated in fit. But, the tree will not be cut. Instead, in the same
        list the "penalty parameter" will be saved in increasing order, _alphas,  the node that will become a leaf will
        be joined, as in a tuple. 

        2- it performs a K-fold on X. After performing the K-fold on X, we will do the same 
        process as we made for the entire X, but on each of those trees. They will also not be cut straightforward.

        3 -  for each alpha in _alphas, and for each K-fold tree, we will search in which interval
        alpha falls in. When we find where it falls, we simply take children_left[node] = -1 and children_right[node] = -1
        for every node in the tuple (alpha, node) in _alpha which is less than or equal to the one from our interval. 

        4 -  after cutting those K-folds, we will test each of them with their fold. Then, take the
        evaluation of the results individually (with self.custom_evaluation). 
        Then, it takes the self.custom_choose of those evaluations and saves it in the error_list. 

        5 -  index with the minimum value, will be the chosen alpha ( and its associated tree). 
        Now that the algorithm knows where to cut the original tree, it will actually cut the tree, by 
        making the children_left[node] = -1 and children_right[node] = -1 for every node in 
        (alpha, node) with alpha <= chosen_alpha. 

        Args:
            X : Matrix of shape (n_samples, n_features).
            Y : Corresponding vector of shape (n_samples,).
        
        """

        alphas_oficial = self._cuts_and_alphas()

        number_splits = min(5,Y.shape[0])
        k_fold = KFold(random_state = 41, k = number_splits)
        k = k_fold.split(X.copy())

        list_test_X =[] 
        list_train_X =[] 
        list_test_y =[] 
        list_train_y =[] 

        list_errors = [] 

        for _, (train, test) in enumerate(k):
            X_train, X_test = X[train], X[test]
            y_train, y_test = Y[train], Y[test]
            list_test_X.append(X_test)
            list_train_X.append(X_train)
            list_test_y.append(y_test)
            list_train_y.append(y_train) 

        trees = [] 
        alphas_list = [] 

        for t in range(len(list_test_X)): 
            tree = TreeBase(min_impurity_decrease = self.min_impurity_decrease, min_samples_split = self.min_samples_split, max_depth = self.max_depth, min_samples_leaf = self.min_samples_leaf, max_leaf = self.max_leaf, random_forest = False,criterion_split = self.criterion_split , custom_split = self.custom_split, criterion_choose = self.criterion_choose , criterion_evaluation = self.criterion_evaluation , custom_evaluation = self.custom_evaluation, factor_selection = self.factor_selection,  form = self.form )#We have to give the same params 
            tree.fit(list_train_X[t], list_train_y[t],prune = False)
            alpha_list_fold = tree._cuts_and_alphas()
            alphas_list.append(alpha_list_fold)
            trees.append(tree)

        for [alpha,node] in alphas_oficial:
            for i in range(len(trees)):
                for ind in range(len(alphas_list[i])):
                    if alphas_list[i][ind][0]<= alpha:
                        if alphas_list[i][ind][1] is not None:
                            trees[i]._change_right_child(alphas_list[i][ind][1],-1)
                            trees[i]._change_left_child(alphas_list[i][ind][1],-1)
                    else:
                        break # Inspect another tree
            #After prunning the trees
            error_of_y_hat = self.custom_choose([self.custom_evaluation(np.array(trees[i].predict(list_test_X[i])),np.array(list_test_y[i])) for i in range(len(trees))])
            list_errors.append(error_of_y_hat)
        list_errors = np.array(list_errors)
        alpha_index = np.argmin(list_errors)
        #definitive prunning
        for ind in range(len(alphas_oficial)):
            if ind <= alpha_index:
                [alpha,node] = alphas_oficial[ind]
                if node is not None:
                    self.children_right[node] = -1 
                    self.children_left[node] = -1 
            else:
                break 
        
    
    def _cuts_and_alphas(self):
        """

        Returns the penalty parameters and associated trees of each.
        Computes the alphas based on the Breiman CART pruning algorithm.


        Returns:
            alpha_list: ordered sequence of the alpha penalties and their nodes that need to be pruned.
        
        """

        def find_N(node):
            if self.children_left[node] != -1:
                self.nodes_below[node] = find_N(self.children_left[node])+find_N(self.children_right[node])
            else:
                self.nodes_below[node] = 1
            return self.nodes_below[node]
        def find_S (node):
            if self.children_left[node] != -1:
                self.S_alg[node] = find_S(self.children_left[node])+find_S(self.children_right[node])
            else:
                self.S_alg[node] = (self.impurity[node])*(self.end[node] - self.start[node])/(self.end[0])
            return self.S_alg[node]
        def find_g(node):
            if self.children_left[node] != -1:
                self.g_small[node] = ((self.impurity[node]*(self.end[node] - self.start[node])/(self.end[0])) - self.S_alg[node])/(self.nodes_below[node] - 1)
                find_g(self.children_left[node])
                find_g(self.children_right[node])
            else:
                self.g_small[node] = np.inf
            return self.g_small[node]

        def find_G(node):
            if self.children_left[node] != -1:

                L = find_G(self.children_left[node])
                R = find_G(self.children_right[node])

                if R[0] > L[0]:
                    self.G_big[node]=[L[0],L[1]] 
                else:
                    self.G_big[node] = [R[0],R[1]]

                if self.G_big[node][0] >self.g_small[node]:
                    self.G_big[node] = [self.g_small[node],node]


            else:
                self.G_big[node] = [np.inf,node]
            return self.G_big[node]

        _ = find_N(0)   
        _ = find_S(0)
        _ = find_g(0)
        _ = find_G(0)

        alpha_list = [[0,None]]

        while self.nodes_below[0] >1:
            if self.G_big[0][0] > max(alpha_list, key = lambda x:x[0])[0]+epsolon:
                alpha_list.append(self.G_big[0])
            copi = self.nodes_below[self.G_big[0][1]] 
            self.nodes_below[self.G_big[0][1]] = 1 
            self.S_alg[self.G_big[0][1]] = (self.impurity[self.G_big[0][1]])*(self.end[self.G_big[0][1]] - self.start[self.G_big[0][1]])/(self.end[0]) 
            self.G_big[self.G_big[0][1]] = [np.inf,self.G_big[0][1]]
            pr_node = self.G_big[0][1]

            while pr_node != 0:
                pr_node= self.father[pr_node]
                self.nodes_below[pr_node] =  self.nodes_below[self.children_left[pr_node]]+self.nodes_below[self.children_right[pr_node]]
                self.S_alg[pr_node] = self.S_alg[self.children_left[pr_node]]+self.S_alg[self.children_right[pr_node]]
                self.g_small[pr_node] = ((self.impurity[pr_node]*(self.end[pr_node] - self.start[pr_node])/(self.end[0])) - self.S_alg[pr_node])/(self.nodes_below[pr_node] - 1)
                L = self.G_big[self.children_left[pr_node]]
                R = self.G_big[self.children_right[pr_node]]
                
                if R[0] > L[0]:
                    self.G_big[pr_node]=[L[0],L[1]] 
                else:
                    self.G_big[pr_node] = [R[0],R[1]]
                
                if self.G_big[pr_node][0] >self.g_small[pr_node]:
                    self.G_big[pr_node] = [self.g_small[pr_node],pr_node]               


         

        return alpha_list




          

class BaggingBase(TreeBase):
    def __init__(self,n_trees = 100, min_impurity_decrease = 0.1, min_samples_split = 2, max_depth = None, min_samples_leaf = 1, max_leaf = None, random_forest = False,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", custom_choose = None, criterion_evaluation = "Standard", custom_evaluation = None,  factor_selection = np.sqrt, form = "regression"):
        """
        Args:
            n_trees: Number of trees in the ensemble (boostrap)
            min_impurity_decrease: The minimum purity gain to allow a split in a node.
            min_samples_split: The minimum number of samples in a node to allow a split.
            max_depth: The maximum depth the tree can grow.
            min_samples_leaf: The minimum number of samples required for each leaf node.
            max_leaf: Maximum number of leaves of the tree. 
            random_forest: Boolean indicating whether it is a random forest or not. 
            criterion_split: String indicating the metric used to calculate the split.
            custom_split: Metric function to calculate the split.
            criterion_choose: String indicating the metric used to choose the predicted value.
            custom_choose: Metric function to choose the predicted value.
            criterion_evaluation: String indicating the metric used to evaluate the error.
            custom_evaluation: Metric function to evaluate the error.
            factor_selection: Function to select the amount of features that can be selected to split a node.
            random_seed: Random seed associated with the tree.
            form: Indicates whether it is a classification or regression tree.
        
        Besides the atributtes associated with the Args, it also  Initializes the attributes:
                
            - self.X_trees_train: List of the feature matrices X used to train each respective bootstrap tree.
            - self.y_trees_train : : List of the vectors y used to train each respective bootstrap tree.
            - self.X_trees_test: : List of the feature matrices X left out during the bootstrap sampling (Out-Of-Bag samples) for each tree.
            - self.y_trees_test : List of the vectors y left out during the bootstrap sampling for each tree.
            - self.list_of_trees: List of train trees chosen by the boostrap (ensemble) function. 
            - self.list_of_indx_test: List of the indices of the Out-Of-Bag (OOB) samples for each respective tree
            - self.qnt_samples: Number of samples of the original matrix X. 
            - self.X_orig : Original matrix X.
            - self.y_orig:Original vector y.
        """
        self.n_trees = n_trees
        if not isinstance(n_trees,int):
            raise TypeError("number of trees must be integer")
        if n_trees < 1:
            raise ValueError("number of trees must be at least one")
        self.X_trees_train =[]
        self.y_trees_train = [] 
        self.X_trees_test =[]
        self.y_trees_test = [] 
        self.list_of_trees = []
        self.list_of_indx_test = [] 
        self.qnt_samples = 0
        self.X_orig =None
        self.y_orig = None
        super().__init__(min_impurity_decrease = min_impurity_decrease , min_samples_split = min_samples_split  , max_depth = max_depth , min_samples_leaf = min_samples_leaf , max_leaf = max_leaf, random_forest = random_forest,criterion_split=criterion_split , custom_split=custom_split , criterion_choose=criterion_choose , custom_choose=custom_choose, criterion_evaluation = criterion_evaluation , custom_evaluation = custom_evaluation,factor_selection = factor_selection, form=form )
    def fit(self, X,y):

        """
        Fits the tree by fitting a tree for each bagging data.

        Args:
            X: Matrix of the observations.
            y: vector of the result. 

        """
        X = np.asarray(X)
        y = np.asarray(y)


        self.X_orig =X
        self.y_orig = y
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain at least one observation and one feature.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The data do not match")

        self.X_trees_train =[]
        self.y_trees_train = [] 
        self.X_trees_test =[]
        self.y_trees_test = [] 
        self.list_of_trees = []
        self.list_of_indx_test = [] 

        self.qnt_samples = X.shape[0]
        resamples = resample(X,n_resamples = self.n_trees,random_state = 42)
        number_of_the_tree = 0 

        for _ ,(train, test) in enumerate(resamples):
            number_of_the_tree+=1
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            tree_test = TreeBase(min_impurity_decrease = self.min_impurity_decrease, min_samples_split = self.min_samples_split, max_depth = self.max_depth, min_samples_leaf = self.min_samples_leaf, max_leaf = self.max_leaf,random_forest=self.random_forest,criterion_split = self.criterion_split , custom_split = self.custom_split, criterion_choose = self.criterion_choose , criterion_evaluation = self.criterion_evaluation , custom_evaluation = self.custom_evaluation,  factor_selection = self.factor_selection, random_seed = number_of_the_tree, form = self.form)
            tree_test.fit(X_train,y_train,prune=False)
            self.list_of_trees.append(tree_test)
            self.X_trees_train.append(X_train)
            self.y_trees_train.append(y_train)
            self.X_trees_test.append(X_test)
            self.y_trees_test.append(y_test)
            self.list_of_indx_test.append(test)
    def predict(self, X_test, trees = None):
        """
        Predicts the y vector associated with the X_test observations. 
        If trees is None, then all trees in the ensemble are used to predict the
        value. The "trees" parameter should only be diffent from "None" if OOB selects a
        smaller amount from self.list_of_trees. 

        Args:
            X_test: Martix of the observations
            trees: trees that should be used to predict
        Returns:
            y_prediction: the vetor y of the predictions associated with the matrix X_test
        """
        X_test = np.asarray(X_test)
        if trees is None:
            trees = self.list_of_trees
        y_preds = [tree.predict(X_test) for tree in trees]
        y_prediction = self.custom_choose(y_preds,axis = 0)
        return np.array(y_prediction)
    def OOB(self):

        """
        Calculates the error Out-of-Bag (OOB)

        Returns:
            The OOB error.
        """
        y_hat_array = [] 
        for n in range(self.qnt_samples):
            tree_list = [] 
            for l in range(len(self.list_of_indx_test)):
                if n in self.list_of_indx_test[l]:
                    tree_list.append(self.list_of_trees[l])
            if len(tree_list) == 0:
                y_hat_array.append(np.nan)
            else:
                y_hat = self.predict(self.X_orig[n:n+1], trees = tree_list)
                value = y_hat[0] if isinstance(y_hat, np.ndarray) and y_hat.size > 0 else y_hat
                y_hat_array.append(value)
        y_hat_array = np.array(y_hat_array, dtype=float)
        y_orig_array = np.array(self.y_orig)
        
        valid = ~np.isnan(y_hat_array)
        if not np.any(valid):
            return "OOB can not be calculated"
        

        return self.custom_evaluation(y_hat_array[valid], y_orig_array[valid])

class RandomForestBase(BaggingBase):
    def __init__(self,n_trees = 100, min_impurity_decrease = 0.1, min_samples_split = 2, max_depth = None, min_samples_leaf = 1, max_leaf = None, random_forest = True,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", custom_choose = None, criterion_evaluation = "Standard", custom_evaluation = None,factor_selection = np.sqrt,form = "regression"):
        """
    RandomForestBase inherits all functionality from BaggingBase, but automatically 
    sets random_forest=True to enable random feature selection at each tree split.
    """
        super().__init__(n_trees = n_trees, min_impurity_decrease = min_impurity_decrease, min_samples_split = min_samples_split , max_depth = max_depth , min_samples_leaf = min_samples_leaf, max_leaf = max_leaf , random_forest = random_forest ,criterion_split = criterion_split, custom_split = custom_split, criterion_choose = criterion_choose , custom_choose = custom_choose, criterion_evaluation = criterion_evaluation , custom_evaluation = custom_evaluation,factor_selection = factor_selection, form = form)






class RegressionTree(TreeBase):
    """
    Inherits all parameters and methods from TreeBase, setting form="regression".
    """
    def __init__(self, min_impurity_decrease = 0.01, min_samples_split = 2, max_depth = None, min_samples_leaf = 1, max_leaf = None, random_forest = False, criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", custom_choose = None, criterion_evaluation = "Standard", custom_evaluation = None , factor_selection = np.sqrt):
        super().__init__( min_impurity_decrease = min_impurity_decrease , min_samples_split = min_samples_split, max_depth = max_depth  , min_samples_leaf = min_samples_leaf , max_leaf =  max_leaf , random_forest =  random_forest, criterion_split = criterion_split, custom_split=custom_split , criterion_choose=criterion_choose , custom_choose=custom_choose , criterion_evaluation = criterion_evaluation, custom_evaluation= custom_evaluation,factor_selection=factor_selection, form = "regression")

class ClassificationTree(TreeBase):
    """
    Inherits all parameters and methods from TreeBase, setting form="classification".
    """
    def __init__(self, min_impurity_decrease = 0.01, min_samples_split = 2, max_depth = None, min_samples_leaf = 1, max_leaf = None, random_forest = False, criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", custom_choose = None,  criterion_evaluation = "Standard", custom_evaluation = None, factor_selection = np.sqrt ):
        super().__init__( min_impurity_decrease =  min_impurity_decrease  , min_samples_split =min_samples_split , max_depth=max_depth , min_samples_leaf=min_samples_leaf , max_leaf= max_leaf , random_forest=random_forest, criterion_split=criterion_split , custom_split=custom_split , criterion_choose =criterion_choose , custom_choose=custom_choose, criterion_evaluation=criterion_evaluation , custom_evaluation=custom_evaluation, factor_selection=factor_selection, form = "classification" )

class BaggingReg(BaggingBase):
    """
    Inherits all parameters and methods from BaggingBase, setting form="regression".
    """
    def __init__(self,n_trees = 100, min_impurity_decrease = 0.1, min_samples_split = 2, max_depth = None, min_samples_leaf = 1, max_leaf = None, random_forest = False,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", custom_choose = None, criterion_evaluation = "Standard", custom_evaluation = None, factor_selection = np.sqrt):
        super().__init__(n_trees=n_trees, min_impurity_decrease=min_impurity_decrease, min_samples_split=min_samples_split, max_depth=max_depth , min_samples_leaf=min_samples_leaf , max_leaf= max_leaf, random_forest=random_forest,criterion_split=criterion_split, custom_split=custom_split, criterion_choose=criterion_choose , custom_choose=custom_choose, criterion_evaluation =criterion_evaluation , custom_evaluation =custom_evaluation, factor_selection=factor_selection, form = "regression")
    def get_params(self):
        """Includes n_trees in get_params()"""
        dic = super().get_params()
        dic['n_trees'] = self.n_trees
        return dic
class Bagging(BaggingBase):
    """
    Inherits all parameters and methods from BaggingBase, setting form="classification".
    """
    def __init__(self,n_trees = 100, min_impurity_decrease = 0.1, min_samples_split = 2, max_depth = None, min_samples_leaf = 1, max_leaf = None, random_forest = False,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", custom_choose = None, criterion_evaluation = "Standard", custom_evaluation = None, factor_selection = np.sqrt):
        super().__init__(n_trees=n_trees, min_impurity_decrease=min_impurity_decrease, min_samples_split=min_samples_split, max_depth=max_depth , min_samples_leaf=min_samples_leaf , max_leaf=max_leaf, random_forest=random_forest,criterion_split=criterion_split, custom_split=custom_split, criterion_choose=criterion_choose , custom_choose= custom_choose, criterion_evaluation=criterion_evaluation , custom_evaluation=custom_evaluation , factor_selection=factor_selection, form = "classification")
    def get_params(self):
        """Includes n_trees in get_params()"""
        dic = super().get_params()
        dic['n_trees'] = self.n_trees
        return dic
    


class RandomForestRegression(RandomForestBase):
    """
    Inherits all parameters and methods from RandomForestBase, setting form="regression".
    """
    def __init__(self,n_trees = 100, min_impurity_decrease = 0.1, min_samples_split = 2, max_depth = None, min_samples_leaf = 1, max_leaf = None, random_forest = True,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", custom_choose = None, criterion_evaluation = "Standard", custom_evaluation = None, factor_selection = np.sqrt):
        super().__init__(n_trees=n_trees, min_impurity_decrease=min_impurity_decrease, min_samples_split=min_samples_split, max_depth=max_depth , min_samples_leaf =min_samples_leaf , max_leaf=max_leaf, random_forest =random_forest,criterion_split=criterion_split, custom_split =custom_split, criterion_choose =criterion_choose , custom_choose = custom_choose, criterion_evaluation =criterion_evaluation , custom_evaluation =custom_evaluation , factor_selection = factor_selection, form = "regression")
    def get_params(self):
        """Includes n_trees in get_params()"""
        dic = super().get_params()
        dic['n_trees'] = self.n_trees
        return dic
class RandomForest(RandomForestBase):
    """
    Inherits all parameters and methods from RandomForestBase, setting form="classification".
    """
    def __init__(self,n_trees = 100, min_impurity_decrease = 0.1, min_samples_split = 2, max_depth = None, min_samples_leaf = 1, max_leaf = None, random_forest = True,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", custom_choose = None, criterion_evaluation = "Standard", custom_evaluation = None, factor_selection = np.sqrt):
        super().__init__(n_trees = n_trees, min_impurity_decrease = min_impurity_decrease, min_samples_split = min_samples_split, max_depth=max_depth , min_samples_leaf = min_samples_leaf , max_leaf=max_leaf, random_forest=random_forest,criterion_split=criterion_split, custom_split= custom_split, criterion_choose=criterion_choose ,custom_choose=custom_choose, criterion_evaluation =criterion_evaluation, custom_evaluation=custom_evaluation , factor_selection=factor_selection, form = "classification")
    def get_params(self):
        """Includes n_trees in get_params()"""
        dic = super().get_params()
        dic['n_trees'] = self.n_trees
        return dic
