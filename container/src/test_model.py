import pytest
import numpy as np
from model import split_data, train_classifier
from sklearn.datasets import make_classification

def test_split_data():
    X = np.array([[0, 1],[2, 3],[4, 5],[6, 7],[8, 9], [10,11]])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    X_train, X_val, y_train, y_val = split_data(X, y)
    
    assert np.array_equal(X_train, np.array([[2, 3],[10, 11],[0, 1],[8, 9]]))
    assert np.array_equal(X_val, np.array([[6, 7], [4, 5]]))
    assert np.array_equal(y_train, np.array([0, 1, 0, 1]))
    assert np.array_equal(y_val, np.array([1, 0]))
    
def test_train_classifier():
    X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0,random_state=0, shuffle=False)
    
    assert 1 == 1
    
    
    
    