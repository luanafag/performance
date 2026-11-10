import numpy as np
from rbg.Utilities import Gini as gini, entropy as ent, classif_error as c_e, mode


from ._RegressionTree import Regression
from ._ClassificationTree import Classification


epsolon = 0.000000000001

class ConstructionTree(Regression, Classification):
    """Base regression class for building a tree and fitting data."""

    def __init__(self,min_impurity_decrease = 0.01, min_samples_split = 2, max_depth = None, min_samples_leaf = 1, max_leaf = None, random_forest = False, criterion_split = "Standard", custom_split = None, criterion_choose = "Standard", custom_choose = None, criterion_evaluation = "Standard", custom_evaluation = None, factor_selection = np.sqrt, random_seed = 0, form = "regression"):
        """Initializes the Tree Structure

                 
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
            form: Indicated whether it is a classification or regression tree.


        Initializes the attributes:
        
            - self.min_impurity_decrease: The minimum purity gain to allow a split in a node.
            - self.min_samples_split: The minimum number of samples in a node to allow a split.
            - self.max_depth: The maximum depth the tree can grow.
            - self.min_samples_leaf: Minimum number of samples required for each leaf node.
            - self.children_left: Array mapping each node to its left child or -1 (if it is a leaf).
            - self.node_depth: Array mapping each node to its depth. 
            - self.children_right: Array mapping each node to its right child or -1 (if it is a leaf).
            - self.feature: Array mapping each node to its split feature index.
            - self.threshold: Array mapping each node to its split threshold value.
            - self.impurity: Array mapping each node to its impurity.
            - self.value: Array mapping each node to its predicted value.
            - self.start: Array mapping each node to the first sample that falls on that node.
            - self.end: Array mapping each node to the last sample that falls on that node.
            - self.aux_purity_gain: Auxiliary array indicating if it is possible to split each node.
            - self.max_leaf: Maximum number of leaves.
            - self.n: Number of nodes.
            - self.random_forest: Boolean indicating if it is a random forest.
            - self.nodes_below: Number of nodes in the subtree of root "i".
            - self.S_alg: Parameter to prune a tree.
            - self.G_big: Parameter to prune a tree.
            - self.g_small: Parameter to prune a tree.
            - self.father: Array mapping each node to its ancestor.
            - self.form: Variable indicating whether it is a classification or regression tree.
            - self.factor_selection: Function that determines the number of predictors to be analyzed.
            - self.random_seed: Random seed associated with the tree, if applicable.
            - self.X: The sample data
            - self.y: The sample data associated with X
            - self.dict_split = {}: Dictionary associating each node with their resulting split parameters.
            - self.max_splits: Maximum number of splits.
        
        Raises:

            TypeError: If the parameter 'min_impurity_decrease' is not a number, or if 
            parameters 'max_depth, min_samples_split,min_samples_leaf.max_leaf 'are not integers

            ValueError: If the integer parameter is smaller than it should be to generate a tree.


        """

        if form == "regression":
            Regression.__init__(self, criterion_split = criterion_split, custom_split =  custom_split, criterion_choose = criterion_choose , custom_choose = custom_choose, criterion_evaluation = criterion_evaluation , custom_evaluation = custom_evaluation )
        else:
            Classification.__init__(self, criterion_split = criterion_split, custom_split = custom_split , criterion_choose = criterion_choose , custom_choose = custom_choose, criterion_evaluation = criterion_evaluation , custom_evaluation =  custom_evaluation )

        if isinstance(min_impurity_decrease, str):
             raise TypeError ("min_impurity_decrease should be a number")        
        if max_depth is not None and not isinstance(max_depth,(int,np.integer)):
            raise TypeError("The param 'max_depth' must be None or an integer.")
        if max_depth is not None and max_depth < 0:
            raise ValueError("'max_depth' must be greater or equal to zero.")
        if  not isinstance(min_samples_split, (int,np.integer)):
            raise TypeError('min_samples_split must be an integer.')
        if min_samples_split < 2:
            raise ValueError ('min_samples_split must be at least 2')
        if not isinstance (min_samples_leaf, (int, np.integer)):
            raise TypeError ("min_samples_leaf should be an integer")
        if min_samples_leaf <1:
            raise ValueError("min_samples_leaf must be at least one")
        if max_leaf is not None and not isinstance (max_leaf, (int, np.integer)):
            raise TypeError("max_leaf must be an integer")
        if max_leaf is not None and max_leaf <1:
            raise ValueError("You must have at least one leaf node")    

        self.min_impurity_decrease = min_impurity_decrease 
        self.min_samples_split = min_samples_split 
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf 
        self.children_left = None
        self.node_depth = None
        self.children_right = None
        self.feature = None 
        self.threshold = None
        self.impurity = None
        self.value = None
        self.start = None
        self.end = None
        self.aux_purity_gain = None
        self.max_leaf = max_leaf
        self.n = None
        self.random_forest = random_forest
        self.nodes_below = None
        self.S_alg = None
        self.G_big = None
        self.g_small = None
        self.father = None
        self.form = form
        self.factor_selection = factor_selection
        self.random_seed = random_seed
        self.X = None
        self.y = None
        self.dict_split = {}

        if max_leaf:
            self.max_splits = max_leaf - 1
        else:
            self.max_splits = None
        
    def _tree_structure(self,X):

        """
        Defines self.n, the number of nodes of the tree, according to the number of samples and previously
        defined limitations (min_samples_leaf, max_depth, and max_leaf). The most restrictive criteria will define
        how many nodes the tree will have.

        Defines the sizes of the arrays  
            - self.children_left
            - self.children_right
            - self.feature 
            - self.node_depth 
            - self.threshold 
            - self.impurity 
            - self.value 
            - self.start 
            - self.end 
            - self.aux_purity_gain 
            - self.nodes_below 
            - self.S_alg
            - self.G_big 
            - self.g_small 
            - self.father 

        Args:
            X: Sample matrix of shape (n_samples, p_feature)
        
        Returns:
            No returns. 
    
        
        """


        if self.min_samples_leaf > np.shape(X)[0]:
            raise ValueError ("The minimum amount of samples leaf is greater than the amount of samples. Either collect more data, or change min_sample_leaf")
        if self.max_depth:
            N = np.shape(X)[0]
            self.n = min(int(2 * np.floor(N/self.min_samples_leaf) -1),int(2 **(self.max_depth +1)) - 1 )
        else:
            N = np.shape(X)[0]
            self.n = 2 * int(np.floor(N/self.min_samples_leaf) )-1 
        if self.max_leaf != None:
            self.n = min(self.n, int(2*self.max_leaf-1)) # The most restrictive condition
        self.children_left = np.ones(self.n,dtype=int) * -1 
        self.children_right = np.ones(self.n,dtype=int) * -1 
        self.feature = np.ones(self.n,dtype=int) * -1 
        self.node_depth = np.ones(self.n,dtype=int) * -1 
        self.threshold = np.full(self.n, np.nan)
        self.impurity = np.ones(self.n) * -1 
        self.value = np.full(self.n, None, dtype = object)
        self.start = np.ones(self.n,dtype=int) * -1
        self.end = np.ones(self.n,dtype=int) * -1
        self.aux_purity_gain = np.ones(self.n,dtype=int) * -1 
        self.nodes_below = np.ones(self.n,dtype=int) * -1 
        self.S_alg= np.ones(self.n,dtype=int) * -1
        self.G_big = np.full(self.n, None, dtype=object)
        self.g_small = np.ones(self.n,dtype=float) * -1.0
        self.father = np.ones(self.n,dtype=int) * -1 
        if self.max_leaf is None:
            self.max_leaf = (self.n+ 1)//2 
            self.max_splits = self.max_leaf - 1

    def _splitter(self,node):

        """

        Designed to find the best predictor and the corresponding best threshold 
        to partition a node. 


        For a fixed node, the algorithm will iterate for every applicable predictor. In each predictor, 
        the method "_best_cut_value_for_fixed_p" will be called to find the best threshold 
        of that feature ( if it exists). Then, the chosen parameter to cut the node will be
        selected. 


        Args:
            node: index of the node that will be tested or actually partitioned
        
        Returns:
            None: If there is no possible split
            float: greatest_purity_gain_overall, the cut that increases the purity gain the most

        """

        ordered_list = [] 
        possible_cuts = [] 

        predictors = int(self.X.shape[1])

        train_idxs = np.arange(predictors)


        if self.random_forest:
            all_idxs = np.arange(predictors)
            seed = (self.random_seed + node) %(2**31) # to ensure that for every tree, and for every node, there is a different seed
            rng = np.random.default_rng(seed)
            predictors = int(self.factor_selection(predictors))
            train_idxs = rng.choice(all_idxs, size=predictors, replace=False)




        for predictor in train_idxs:

            lines = self.X[self.start[node]:self.end[node]] #lines from the sample that are on the current node
            ordered_ind = np.argsort(lines[:,predictor]) # increasing order index of those lines according to the predictor
            final_lines = lines[ordered_ind] # re - arranged lines that are on "node" but in increasing order according to predictor
            col_pred = final_lines[:, predictor]


            y_lines = self.y[self.start[node]:self.end[node]]
            final_y_lines = y_lines[ordered_ind]

            thresholds = (col_pred[:-1] + col_pred[1:])/2 # Candidates for cut value. 

            self._best_cut_value_for_fixed_p(node,thresholds,ordered_list,possible_cuts,final_y_lines,predictor,final_lines,ordered_ind)


        ordered_list = np.array(ordered_list)
        possible_cuts = np.array(possible_cuts)



        if ordered_list.shape[0] == 0:
            return None 
        else:
            greatest_purity_gain_overall_index = np.argmax(possible_cuts[:,0])
            greatest_purity_gain_overall = possible_cuts[greatest_purity_gain_overall_index,0]

            self.dict_split[node] = [ greatest_purity_gain_overall_index, possible_cuts, ordered_list]
            
            return greatest_purity_gain_overall


    def _best_cut_value_for_fixed_p(self,node,thresholds,ordered_list,possible_cuts,final_y_lines,predictor,final_lines,ordered_ind):
            """

            For a given predictor and node, the module finds that best cut value ( if it exists), and:
                - add to 'matri_list' the reordered X matrix after the potential split (X_c_p).
                - add to 'y_list' the reordered y vector to match X_c_p.
                - add to 'possible_cuts' the best cut found, which is a list containing:
                    * greatest_purity_gain_per_p: The best purity gain, over all cut values for this fixed predictor.
                    * associated_value_per_p: The value cut that maximized the purity gain.
                    * predictor: The fixed predictor index.
                    * new_end_begin: The index delimiting the new "slice" of samples that belongs to the left child and right child. 
                    * imp_left: impurity of the left child.
                    * imp_right: impurity of the right child.
                    * prev_l: prediction of the left child.
                    * prev_r: prediction of the right child.
            
            If the split violates any of the conditions 
                -'The minimum purity gain to allow a split in a node.'
                -'Minimum number of samples of each leaf.'
            This value cut will be discarded. 
            If all of the cut values are invalid, then 'greatest_purity_gain_per_p' will remain None,
            and the fixed predictor will not be considered to partition the node. 


            Args:
                - node: node evaluated for the split
                - thresholds: sequence of the possible cut values. 
                - ordered_list: list of the ordered lines of the cut.
                - possible_cuts: list of the possible cuts of the node
                - final_y_lines: rearranged lines from the vector y but in increasing order according to predictor, to match the X ones.
                - predictor: the fixed predictor
                - final_lines: rearranged lines from matrix X that are on "node" but in increasing order according to predictor
                - ordered_ind: Ordered sequence that determines final_lines and final_y_lines.
            
            
            """


            m = 0
            greatest_purity_gain_per_p = None
            associated_value_per_p = None
            new_end_begin = None
            imp_left = None
            imp_right = None
            prev_l= None
            prev_r = None
            while m < len(thresholds):
                id_last_m = np.searchsorted(thresholds, thresholds[m], side = 'right') - 1  # Since we are using a sorted array, we can use a function that uses binary search to avoid recounting repeated means.
                if thresholds[m] == thresholds[id_last_m]:
                    m = id_last_m 
                if thresholds[m] == final_lines[m+1,predictor]: # the scenario that two sorted samples have the same predictor value. The mean of those two is the same as their value.
                    if m+1 == len(thresholds): #Extremly important break. If it is true, then it means that we would try to split by letting 
                        break # everything on the left child. Besides being prohibited, it would cause index error in the next lines.
                    else: # if m+=1 is not out of range:
                        m+=1
                
                if m+1 >=  self.min_samples_leaf and len(thresholds)+1 - (m +1) >= self.min_samples_leaf: # checking if the left child and right child both have the minimum amount of samples
                    y_left_slice = final_y_lines[:m + 1]
                    y_right_slice = final_y_lines[m + 1:]
                
                    prevision_left = self.custom_choose(y_left_slice) # add +1 after m becuase in sclicing the end is exclusive
                    prevision_right = self.custom_choose(y_right_slice) 
                    left_impurity = self.custom_split(np.array(prevision_left),np.array(y_left_slice)) 
                    right_impurity = self.custom_split(np.array(prevision_right),np.array(y_right_slice))
                    current_purity_gain = self.impurity[node] - ((m+1)/ (len(thresholds) +1)* left_impurity + ((len(thresholds)+1)-(m+1))/(len(thresholds)+1)*right_impurity)



                    if current_purity_gain >= self.min_impurity_decrease: # Checking if the purity gain is equal or greater than the minimum necessary
                        if  greatest_purity_gain_per_p is not None: #If a value was already computed, we have to check if the new one is better
                            if greatest_purity_gain_per_p < current_purity_gain:
                                greatest_purity_gain_per_p = current_purity_gain
                                associated_value_per_p = thresholds[m]
                                new_end_begin = self.start[node] +m+1 # considering [start , end)
                                imp_left = left_impurity
                                imp_right = right_impurity
                                prev_l= prevision_left
                                prev_r = prevision_right
                        else: # If no value was computed, the current is the best
                            greatest_purity_gain_per_p = current_purity_gain
                            associated_value_per_p = thresholds[m]
                            new_end_begin = self.start[node] +m+1 # considering [start , end)
                            imp_left = left_impurity
                            imp_right =right_impurity
                            prev_l= prevision_left
                            prev_r = prevision_right
                m +=1
            if greatest_purity_gain_per_p is not None:
                ordered_list.append(ordered_ind)
                possible_cuts.append([greatest_purity_gain_per_p,associated_value_per_p,predictor,new_end_begin,imp_left, imp_right,prev_l,prev_r])
    def _tree_construction(self, X,y):

        """

        Builds a decision tree from the matrix X and vector y. 

        It calls the _tree_structure to initialize the tree structure and sets up the 
        root node (0). Then, it evaluates whether it is available to cut the root
        according to the predefined tree constraints. 
        
        It uses the function "_splitter" to find the best split of the root, if it is applicable. If "_splitter" 
        finds a good split, and if there is still available splits, the node will be split. This process
        will continue in loop until there is no more splits available and while "_splitter" still finds available
        partitions. The node that provides the greatest purity gain at each step of the algorithm that will
        be split, according to the logic of "best-first growing strategy".

        It uses self.dict_split to acess the informations of the node selected to be split, to actually split it. 
        

        Args:
            X: Matrix of shape (n_samples, p_features)
            y: Vector of the observations of shape (n_samples,).
        
        Raises:
            ValueError: if the number of dimensions of X and y are incompatible

        """
        if X.ndim != 2:
            raise ValueError("X must be two-dimensional")
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain at least one observation and one feature.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The data do not match")
        self._tree_structure(X)
        self.X = np.asarray(X.copy())
        self.y = np.asarray(y.copy())
        self.value[0] = self.custom_choose(y) 
        self.impurity[0] = self.custom_split(self.value[0],self.y) 
        self.start[0] = 0
        self.end[0] = self.X.shape[0] # considering [start , end)
        self.node_depth[0] = 0

        list_nodes_that_can_be_split =  [] 

        if (self.max_leaf >1 or self.max_leaf is None) and  self.X.shape[0] >=  self.min_samples_split and (self.max_depth is None or self.max_depth >=1 ) and self.impurity[0]>epsolon:
            gain = self._splitter(0)
            if gain:
                list_nodes_that_can_be_split.append([gain,0])
                self.aux_purity_gain[0] = 1

    
        while ( 1 in self.aux_purity_gain and self.max_splits>0):
            if len(list_nodes_that_can_be_split) >0:
              
                chosen = max(list_nodes_that_can_be_split, key = lambda x:x[0]) 
                node_chosen_to_split = chosen[1]
                list_nodes_that_can_be_split.remove(chosen)
              
                self.aux_purity_gain[node_chosen_to_split] = -1 # turning it back to -1 

                greatest_purity_gain_overall_index = self.dict_split[node_chosen_to_split][0] 
                possible_cuts = self.dict_split[node_chosen_to_split][1] 
                ordered_list = self.dict_split[node_chosen_to_split][2]
                del self.dict_split[node_chosen_to_split]

                lines = self.X[self.start[node_chosen_to_split]:self.end[node_chosen_to_split]] #lines from the sample that are on the current node
                ordered_ind = ordered_list[greatest_purity_gain_overall_index] # increasing order index of those lines according to the predictor
                final_lines = lines[ordered_ind] # re - arranged lines that are on "node" but in increasing order according to predictor

                self.X[self.start[node_chosen_to_split]:self.end[node_chosen_to_split]] = final_lines # re - arrenged matrix X 
                y_lines = self.y[self.start[node_chosen_to_split]:self.end[node_chosen_to_split]]
                y_final_lines = y_lines[ordered_ind]
                self.y[self.start[node_chosen_to_split]:self.end[node_chosen_to_split]] = y_final_lines
                self.threshold[node_chosen_to_split] = possible_cuts[greatest_purity_gain_overall_index,1]
                self.feature[node_chosen_to_split] = possible_cuts[greatest_purity_gain_overall_index,2]
                r1 = np.max(self.children_right)
                r2 = np.max(self.children_left)
                right = max(r1,r2)+1
                left = right+1
                if right == 0: #because in the first split, all values will be -1, but 0, 0 = -1 +1, is the root.
                    right =1 
                    left = 2         

                self.children_left[node_chosen_to_split] = left
                self.children_right[node_chosen_to_split] = right

                self.value[left] = possible_cuts[greatest_purity_gain_overall_index,6]
                self.impurity[left] = possible_cuts[greatest_purity_gain_overall_index,4]
                self.start[left] = self.start[node_chosen_to_split]
                self.end[left] = possible_cuts[greatest_purity_gain_overall_index,3]
                self.node_depth[left] = self.node_depth[node_chosen_to_split] +1 
                self.father[left] = node_chosen_to_split

                if (self.end[left] - self.start[left]) >= self.min_samples_split and (self.max_depth is None or self.node_depth[left] < self.max_depth) and self.impurity[left] > epsolon:
                    self.aux_purity_gain[left] = 1

                self.value[right] = possible_cuts[greatest_purity_gain_overall_index,7]
                self.impurity[right] = possible_cuts[greatest_purity_gain_overall_index,5]
                self.start[right]= possible_cuts[greatest_purity_gain_overall_index,3] 
                self.end[right] = self.end[node_chosen_to_split]
                self.node_depth[right] = self.node_depth[node_chosen_to_split] +1 
                self.father[right] = node_chosen_to_split

                if (self.end[right] - self.start[right]) >= self.min_samples_split and (self.max_depth is None or self.node_depth[right] < self.max_depth) and self.impurity[right] > epsolon:
                    self.aux_purity_gain[right] = 1
                    

                self.max_splits -=1

                left_child = self.children_left[node_chosen_to_split]
                right_child = self.children_right[node_chosen_to_split]

                if self.aux_purity_gain[left_child] == 1: 
                    new_candidate = self._splitter(left_child)
                    if new_candidate: # That is, if it returned a value different from "None" and gain > 0
                        list_nodes_that_can_be_split.append([new_candidate,left_child])

                    else: # if in reality, we can not split it
                        self.aux_purity_gain[left_child] = -1 
                if self.aux_purity_gain[right_child] == 1: 
                    new_candidate = self._splitter(right_child)
                    if new_candidate:# That is, if it returned a value different from "None" and gain > 0
                        list_nodes_that_can_be_split.append([new_candidate,right_child])

                    else:
                        self.aux_purity_gain[right_child] = -1

    def predict (self,X):

        """
        Predict X after the tree has been built.

        Args:
            X: Sample matrix of shape (n_samples, p_features) or a 1D array for a single sample.
        
        Returns:
            y: np.ndarray of the predicted values of shape (n_samples,).
        
        """

        if X.ndim == 1:
            X= X.reshape(1,-1)

        y_num = X.shape[0]
        
        y = [] 
        for i in range(y_num):
            node = 0
            while self.children_left[node] != -1: # If it has left child, then it is an internal node (or the root)
                if X[i][self.feature[node]] <= self.threshold[node]:
                    node = self.children_left[node]
                else:
                    node = self.children_right[node]
            pred_y = self.value[node]
            y.append(pred_y)
        y = np.array(y)
        return y
    def set_params(self, **params):
        """
        Sets the parameters of the decision tree.

        This method updates only the specified parameters without overwriting 
        previously configured values for other parameters.

        Args:
            Parameters that need to be changed and the new value.
        Raises:
            ValueError if an invalid parameter is submitted.

        """
        if not params:
            pass
        else:
            defined_params = self.get_params()

            for key, value in params.items():
                if key in defined_params:
                    defined_params[key] = value
                else:
                    raise ValueError("Invalid parameter")
            self.__init__(**defined_params)

        


    def get_params(self):
        """

        Function to get the parameters of the tree in a dictionary.

        Returns:
            Dictionary of the parameters.
        
        """
        dicto = {
            "min_impurity_decrease": self.min_impurity_decrease,
            "max_depth":self.max_depth ,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_leaf": self.max_leaf ,
            "random_forest": self.random_forest,
            "criterion_split": self.criterion_split,
            "custom_split": self.custom_split,
            "criterion_choose": self.criterion_choose,
            "custom_choose": self.custom_choose,
            "criterion_evaluation":self.criterion_evaluation,
            "custom_evaluation": self.custom_evaluation,
            "factor_selection": self.factor_selection
            }
        return dicto
    def get_tree_structure_info(self):

        """
        Function to get the tree structure in a dictionary.

        Returns: 
            Dictionary of the tree structure.

        """
        dicto = {
        "children_left":self.children_left,
        "children_right":self.children_right,
        "feature":self.feature,
        "node_depth":self.node_depth ,
        "threshold":self.threshold,
        "impurity":self.impurity,
        "value":self.value,
        "start":self.start,
        "end":self.end,
        "aux_purity_gain":self.aux_purity_gain,
        "n": self.n
            }
        return dicto
    def _change_right_child(self,father,new_value):
        """
        Internal function to change the correspondence between a node 
        and its right child.

        Args:
            father: Index of father node.
            new_value: Index of new right child.
        
        """
        self.children_right[father] = new_value
    def _change_left_child(self,father,new_value):

        """
        Internal function to change the correspondence between a node 
        and its left child.

        
        Args:
            father: Index of father node.
            new_value: Index of the new left child.
                
        """
        
        self.children_left[father] = new_value
    
