"""
My humble contribution of generic ML utils to support my own projects.

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.utils import check_array
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score
)
from sklearn.model_selection import KFold, RepeatedKFold, cross_validate

# this is mine - one needs to
# pip install git+https://github.com/gerberl/disfunctools.git
from disfunctools import (
    pairwise_lst_el_combinations, pairwise_lst_el_chain,
    pairwise_lst_1st_vs_rest
)
from functools import partial
import re


# ## Data normalisation


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



# ## Those itertools/functools/numpy utility functions for generating pairs:
#    here, elements are columns of X

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




# ## Model Evaluation

def get_names_func_eval_metrics():
    return dict(
        MAE=mean_absolute_error,
        MdAE=median_absolute_error,
        RMSE=root_mean_squared_error,
        MAPE=mean_absolute_percentage_error,
        r2=r2_score
    )

def get_names_of_eval_scores():
    return dict(
        nMAE='neg_mean_absolute_error',
        nMdAE='neg_median_absolute_error',
        nRMSE='neg_root_mean_squared_error',
        nMAPE='neg_mean_absolute_percentage_error',
        r2='r2'
    )



def cross_validate_it(
        reg, X, y, 
        scoring=get_names_of_eval_scores(),
        cv=KFold()
    ):
    """
    e.g.,

    >>> cv_results_df = cross_validate_it(reg, X, y)
    >>> cv_results_df

                    mean       std
    test_MAE    0.369914  0.056890
    train_MAE   0.362901  0.014563
    test_RMSE   0.450043  0.049646
    train_RMSE  0.445522  0.012497
    test_MAPE   1.246502  0.785347
    train_MAPE  1.208433  0.155948
    test_r2     0.034445  0.090380
    train_r2    0.101184  0.019556
    """

    cv_results = cross_validate(
        reg, X, y, scoring=scoring, cv=cv, return_train_score=True
    )
    
    # from cv_results, I would like to keep all scores/losses, but wouldn't
    # need any time-related metrics here
    scores_dict = { k: v for k, v in cv_results.items() if 'time' not in k }
    scores = pd.DataFrame(scores_dict)
    # all negative scores (i.e., losses) have their sign reverse (i.e., *-1)
    # asumption is that they would have been given a prefix `_n`
    # the sign-reversed ones are concatenated with those passed through
    scores = pd.concat(
        [ 
            scores.filter(regex='_n', axis=1)*-1, 
            scores.drop(
                columns=scores.filter(regex='_n', axis=1).columns, axis=1
            )
        ], 
        axis=1
    )

    # keep only the main statistics of the distribution of the scores
    score_stats = pd.DataFrame(
        dict(mean=scores.mean(), std=scores.std())
    )
    # rename negative scores now, as they are shown as losses (just remove _n)
    score_stats = score_stats.set_axis(
        score_stats.index.str.replace('_n', '_'), axis=0
    )

    return score_stats



def pivot_cross_validated_stats(df):
    """
    https://chatgpt.com/c/dbc13193-7e80-4ad8-91e1-5b041b556d6e
    e.g.,

    >>> pivot_cv_results_df = pivot_cross_validated_results(cv_results_df)
    >>> pivot_cv_results_df

            test_mean  train_mean  test_std  train_std
    metric
    MAE      0.369914    0.362901  0.056890   0.014563
    MAPE     1.246502    1.208433  0.785347   0.155948
    RMSE     0.450043    0.445522  0.049646   0.012497
    r2       0.034445    0.101184  0.090380   0.019556
    """

    # Reset the index to make it a column (assuming it is the metric name)
    df = df.reset_index()
    # Split the index column into 'set' and 'metric'
    df[['set', 'metric']] = df['index'].str.split('_', expand=True)
    # Drop the original 'index' column
    df = df.drop(columns=['index'])
    # Pivot the table
    pivot_df = df.pivot(index='metric', columns='set')
    # Flatten the MultiIndex columns
    pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
    pivot_df = pivot_df.reset_index()
    # Rename columns for clarity
    pivot_df.columns = ['metric', 'test_mean', 'train_mean', 'test_std', 'train_std']
    # Display the reshaped DataFrame
    return pivot_df.set_index('metric')



def get_CV_train_test_scores(
        reg, X, y, 
        scoring=get_names_of_eval_scores(),
        cv=KFold()
    ):

    cv_results_df = cross_validate_it(reg, X, y, cv=cv)
    cv_results_df = pivot_cross_validated_stats(cv_results_df)
    # I fancy this order of metrics
    cv_results_df = cv_results_df.reindex(
        ['MAE', 'MdAE', 'RMSE', 'MAPE', 'r2']
    )
    return cv_results_df



def plot_true_vs_predicted(
        est, 
        X_train, y_train,
        X_test, y_test,
        ax=None,
        train_style_kws={},
        test_style_kws={}
    ):
    """
    A few strong assumptions here (e.g., that there is always a test set).
    Could do with some refactoring, but OK for the moment.
    """
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)

    y_pred_train = est.predict(X_train)
    y_pred_test = est.predict(X_test)
    ax.plot(y_train, y_pred_train, '.', label='train', **train_style_kws)
    ax.plot(y_test, y_pred_test, '.', label='test', **test_style_kws)
    ax.set_xlabel('True Target')
    ax.set_ylabel('Predicted Target')

    # need to make both axis have the same (x,y)-limits
    all_target_values = np.concatenate([y_pred_train, y_pred_test, y_train, y_test])
    min_target = min(all_target_values)
    max_target = max(all_target_values)
    target_lim = (min_target, max_target)
    ax.set_xlim(target_lim)
    ax.set_ylim(target_lim)

    # "nudge" extremes slightly so that all data points are visible
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_margin = (xlim[1] - xlim[0]) * 0.01  # 5% margin
    y_margin = (ylim[1] - ylim[0]) * 0.01  # 5% margin
    ax.set_xlim(xlim[0] - x_margin, xlim[1] + x_margin)
    ax.set_ylim(ylim[0] - y_margin, ylim[1] + y_margin)    

    # the diagnonal line for the idealised space of predictions
    ax.plot(
        [0, 1], [0, 1], transform=ax.transAxes, 
        color='gray', linestyle=':', alpha=0.3
    )
    ax.legend()

    return ax



def plot_rf_feat_imp_barh(rf, feat_names, ax=None, top_feat_k=10, style_kws={}):
    """ """
    if ax is None:
        fig, ax = plt.subplots()
    
    return pd.Series(
        rf.feature_importances_, 
        index=feat_names
    ).sort_values().tail(top_feat_k).plot.barh(**style_kws)



# ## Hyper-Parameter Search

# def refit_strategy(cv_results):
#     """
#     9-Sep-24: I don't think I need this!

#     To ensure that the best model is that with the best test score, I've
#     found adding `refit=refit_strategy` to my hyps enough to do the job:

#     Example:
#     grid_search = GridSearchCV(
#         estimator=polyb, param_grid=param_grid, cv=5, return_train_score=True,
#         refit=refit_strategy
#     )
#     """
#     df = pd.DataFrame(cv_results)
#     df = df.sort_values(by='mean_test_score', ascending=False)
#     return df.head(1).index.to_numpy().item()

def format_cv_results(grid_search, verbose=False, train_score=True, sort_by_rank=True):
    """
    Return a DataFrame with GridSearchCV results (and the like) in a format that
    I prefer; Only parameters, optionally train and test score means and
    standard deviations, and rank based on test score. `param` and
    (optionally) pipeline prefixes are removed.
    """
    cv_results_full = pd.DataFrame(grid_search.cv_results_)

    cols = [ ]
    cols += cv_results_full.filter(regex=r'param_').columns.tolist()
    
    score_features = [ ]
    if train_score:
        score_features += [ 'mean_train_score', 'std_train_score' ]
    score_features += [ 'mean_test_score', 'std_test_score' ]
    score_features += ['rank_test_score']
    
    cols += score_features

    hs_results = cv_results_full[cols]

    # hopefully this removes the pipeline column name prefixes if asked
    if not verbose:
        hs_results = hs_results.rename(
            columns=lambda col: re.sub('.+__', '', col)
        )

    if sort_by_rank:
        hs_results = hs_results.sort_values('rank_test_score', ascending=True)
    
    return hs_results


def format_optuna_hs_results(
        optuna_search, verbose=False, train_score=True, sort_by_rank=True
    ):

    hs_results_full = pd.DataFrame(optuna_search.trials_dataframe())

    cols = [ ]
    cols += hs_results_full.filter(regex=r'params_').columns.tolist()

    score_features = [ ]
    if train_score:
        score_features += [ 
            'user_attrs_mean_train_score', 'user_attrs_std_train_score'
        ]
    score_features += [ 
        'user_attrs_mean_test_score', 'user_attrs_std_test_score'
    ]
    score_features += ['value']
    
    cols += score_features

    hs_results = hs_results_full[cols]

    # some more managable column names
    hs_results = (hs_results
        .rename(columns=lambda col: re.sub('user_attrs_', '', col))
        .rename(columns=lambda col: re.sub('params_', '', col))
    )

    # hopefully this removes the pipeline column name prefixes if asked
    if not verbose:
        hs_results = hs_results.rename(
            columns=lambda col: re.sub('.+__', '', col)
        )

    if sort_by_rank:
        hs_results = hs_results.sort_values('mean_test_score', ascending=False)
    
    return hs_results
    
