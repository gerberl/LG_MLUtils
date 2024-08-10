"""
My humble contribution of generic ML utils to support my own projects.

"""
import numpy as np
from sklearn.utils import check_array

# this is mine - one needs to
# pip install git+https://github.com/gerberl/disfunctools.git
from disfunctools import (
    pairwise_lst_el_combinations, pairwise_lst_el_chain,
    pairwise_lst_1st_vs_rest
)



def min_max_norm(x):
    """could rely on sklearn's implementation, but a bit of faffing involved
    with X and y shapes"""

    x = check_array(x, ensure_2d=False)
    x = (x - x.min()) / (x.max() - x.min())
    return x

def zscore_norm(x):
    """could rely on sklearn's implementation, but a bit of faffing involved
    with X and y shapes"""

    x = check_array(x, ensure_2d=False)
    x = (x - x.mean()) / (x.std())
    return x


# ## Utility functions for generating pairs: here, elements are columns of X

def pairwise_X_cols_combinations(X):
    """
    >>> X = np.random.randint(1, 10, (4, 5))
    >>> X
    array([[4, 2, 7, 7, 7],
           [2, 2, 6, 3, 1],
           [9, 4, 6, 7, 7],
           [2, 9, 7, 8, 3]])
    >>> list(pairwise_X_cols_combinations(X))
    [
        (array([4, 2, 9, 2]), array([2, 2, 4, 9])),
        (array([4, 2, 9, 2]), array([7, 6, 6, 7])),
        (array([4, 2, 9, 2]), array([7, 3, 7, 8])),
        (array([4, 2, 9, 2]), array([7, 1, 7, 3])),
        (array([2, 2, 4, 9]), array([7, 6, 6, 7])),
        (array([2, 2, 4, 9]), array([7, 3, 7, 8])),
        (array([2, 2, 4, 9]), array([7, 1, 7, 3])),
        (array([7, 6, 6, 7]), array([7, 3, 7, 8])),
        (array([7, 6, 6, 7]), array([7, 1, 7, 3])),
        (array([7, 3, 7, 8]), array([7, 1, 7, 3]))
    ]
    """
    return pairwise_lst_el_combinations(X.T)

def pairwise_X_cols_chain(X):
    """
    >>> X = np.random.randint(1, 10, (4, 5))
    >>> X
    array([[4, 2, 7, 7, 7],
           [2, 2, 6, 3, 1],
           [9, 4, 6, 7, 7],
           [2, 9, 7, 8, 3]])
    >>> list(pairwise_X_cols_chain(X))
    [
        (array([4, 2, 9, 2]), array([2, 2, 4, 9])),
        (array([2, 2, 4, 9]), array([7, 6, 6, 7])),
        (array([7, 6, 6, 7]), array([7, 3, 7, 8])),
        (array([7, 3, 7, 8]), array([7, 1, 7, 3]))
    ]
    """
    return pairwise_lst_el_chain(X.T)


def pairwise_X_cols_1st_vs_rest(X):
    """
    >>> X = np.random.randint(1, 10, (4, 5))
    >>> X
    array([[4, 2, 7, 7, 7],
           [2, 2, 6, 3, 1],
           [9, 4, 6, 7, 7],
           [2, 9, 7, 8, 3]])
    >>> list(pairwise_X_cols_1st_vs_rest(X))
    [
        (array([4, 2, 9, 2]), array([2, 2, 4, 9])),
        (array([4, 2, 9, 2]), array([7, 6, 6, 7])),
        (array([4, 2, 9, 2]), array([7, 3, 7, 8])),
        (array([4, 2, 9, 2]), array([7, 1, 7, 3]))
    ]
    """
    return pairwise_lst_1st_vs_rest(X.T)


