import numpy as np
from rbg.Utilities import Gini as gini, entropy as ent, classif_error as c_e, mode, error_rate

def Gini(_, y_true):
    return gini(y_true)
def entropy(_, y_true):
    return ent(y_true)
def classif_error(_, y_true):
    return c_e(y_true)

epsolon = 0.000000000001

class Classification:
    '''

    Initializes the metrics used to calculate the impurity (that is, the metrics to split the nodes),
    predict and evaluate. For the impurity, it allows the user to use Gini as default, but also allows the classification
    metrics implemented by the library. The user an also define a function of his choice, however,
    it must take as parameters, y_predicted and y_true, respectively.
    For the predict, it allows median, max, min, mode, or another one chosen by the
    user. However, the user must define and submit a function that accepts an array as a paramter, and 
    "axis", if the user is performing bagging, random forest or prunning a tree. 
    The evaluation function is by standard "Standard", to indicate that is the error rate. 
    The user is allowed to choose another one that must take as parameters, y_predicted and y_true, respectively.

    Args:

        criterion_split: string defining what will be the function to calculate the impurity (if "Other", then the user must provide a function)
        custom_split: the function itself    
        criterion_choose: string defining what will be the function to select the y_predic value (if "Other", then the user must provide a function)
        custom_choose: the function itself (None if the user choose one of the median, min, max or mode) 
        criterion_evaluation: string defining what will be the function to calculate the evluation / error (if "Other", then the user must provide a function)
        custom_evaluation: the function itself 

    Raises:

        ValueError: if any of the strings from criterion_split, criterion_choose or criterion_evaluation are invalid strings
        Warning:If the user chooses a different metric for the evaluation from the impurity metric. 
    '''
    def __init__(self,criterion_split = "Standard", custom_split = None, criterion_choose = "Standard",custom_choose = None, criterion_evaluation = "Standard", custom_evaluation = None):
        split ={
            'Standard':Gini,
            'Gini':Gini,
            'classif_error':classif_error,
            'entropy':entropy,
            'Other':custom_split
        }
        if criterion_split not in split:
            raise ValueError ("The only split criterions available for classification are Gini,classif_error,entropy, and Other")
        self.custom_split  = split.get(criterion_split)

        y_pred ={
            'Standard': mode,
            'median':np.median,
            'max':np.max,
            'min':np.min,
            'mode':mode,
            'Other':custom_choose 
        }
        if criterion_choose not in y_pred:
            raise ValueError ("The only split criterions available for assinging y_pred are mode, median, min, and max")
        self.custom_choose  = y_pred.get(criterion_choose)
        evalu ={
            'Standard':error_rate,
            'Same':self.custom_split,
            'Other':custom_evaluation
        
        } 

        if criterion_evaluation not in evalu:
            raise ValueError ("The only evaluation criterions available for evaluating the tree are Gini, entropy, criterion_split, the Same as used in spliting and Other")
        self.custom_evaluation  = evalu.get(criterion_evaluation)
        if criterion_evaluation !="Standard":
            raise Warning("It is possible, but probably inefficient to assaing a different method of error rate")

        self.criterion_split = criterion_split
        self.criterion_choose = criterion_choose
        self.criterion_evaluation = criterion_evaluation
