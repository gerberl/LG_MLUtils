* some key points of the last week or so:
    - a functional, multivariate PolyRegr (mainly Chebyshev Huber, but would also like to try BSplies)
    
    - more robust approach to tuning (hyper-parameter search); tried the usual GridSearch and RandomizedSearch, but also the supposedly more sophisticated Ray Tune and optuna; defined search space defined in terms of distributions (rather than simply numpy arrays).
    
    - OutlierRemoval (feature_engine)
    
    - TransformedTargetRegressor (wrapper to log/exp target)
    
    - Traning/fitting/optimising on metrics less sensitive to outliers (e.g., R^2, MAE); how about those pinball and the like, the one used by Huber
    

## PolyRegr Chebyshev Huber

* I think I have a working implementation now for multivariate data.

* I have only tried on AMES data, using numeric features only (a mixture of discrete and continuous ones).

* After some wrestling, it turns out that `optuna` does seem to be a better tuner; it did seem to converge to a region of good hyper-parameters. I am going ahead with it for our Chebyshev Complexity metric:

```python
param_dist = dict(
    est__alpha=np.logspace(-4, +4, 9),
    est__epsilon=np.linspace(1.05, 2.55, 6),
    est__complexity=np.linspace(2, 10, 9, dtype=int)
)
```

* ChebHuber seems to start overfitting quite quickly; also, with some occasional really large residuals.

* I had a look at removing outliers with `feature_engine`'s `OutlierRemover`. It did improve the performance of the ChebHuber with regards to the occasional large residual. However, specially for my benchmarks, I much rather not use it but instead rely on metrics that are not as affected by outliers (e.g., R^2, MdAE); also, when plotting, say, true vs predicted or SHAP scatter plots, I should then do some filtering of outliers so that the visualisations are meaningful.

* The occasional large residuals can cause overflow if I am using metrics or transformations that involve squaring and exponentiating; I faced this using MSE and also the `np.log1p` and `np.expm1` transforms. At this stage, I'll probably try to stick to r2, MAE, or MdAE (as long as they are differentiable and suitable for the underlying optimisation).

* The issue with the large residuals reminds me that ERFBN suffered from too; should be useful for analyses for out benchmark paper.

* Where I am then with regards to a general purpose PolyRegr?
    - I think I need to implement `get_features_out` or the like for making it work seemlessly with pipelines when output is set to pandas. This is the warning message I get - it could be that all I need is to implement `get_feature_names_out` or the like for the predicted `y`:
    
    ```
    UserWarning: When `set_output` is configured to be 'pandas', `func` should return a pandas DataFrame to follow the `set_output` API  or `feature_names_out` should be defined.
    ```

    - For now, I think I am good, but perhaps it would be a good idea to have a custom transformer that wraps around the generic vander functions (those that generate the design matrices) and the pre-processing functionality; would need to then name polynomial features accordingly (need to check the documentation of each vander). 
    
    - With the item above implemented, I can do some refactoring so that PolyRegr is really just a pipeline wrapping around the PolyTransformer and the Huber regressor. Again, right now, I can live with how it is.

    - I would like to try the refactored PolyRegr on my XAI datasets again (Drawn and AMES SHAP 2D arrays) and see how the Chebyshev complexity metric and plots behave.
    
- SHAP sensitivity analysis on ChebHuber on the AMES example (e.g., `cheb-huber-optuna-hs-ames-12Sep24`) looks promising in terms of XAI. Try also on ERBFN soon.

- Bear in mind that my PolyRegr has a `normalise` functionality built-in: it scales the data (the X) to the interval the polynomial type is defined (e.g., Chebyshev is [-1, 1]). See `MinMaxScaler(feature_range=norm_interval))`.



## ERBFN benchmark work/paper

* I have made the decision of refactoring so as to reflect better our updated understanding of ML workflows, tuning, and evaluation. Things have moved on since we started circa 2020. There have been some work works on tabular regression and NN, and we will mention something about it.

* Try to keep it to a bare minimum; it could easily become a rabbit hole. 

* We could use the **latest version of regression datasets in PMLB**
    - https://github.com/EpistasisLab/pmlb
    - I might use the pmlb package and query the regression datasets easily, I think; Otherwise, there is a tsv file to explore if needed
    - Again, there might be some exclusion criteria (e.g., too large cardinality/arity).

* To prevent issues with too many features, perhaps do **TargetEncoding**? Test the `sklearn` one again; if not clear, back to the `category_encoders` one.

  
* Fix (do I need to?) the early termination on validation set.
      
* Have the adjusted R^2 and the median absolute deviation when reporting - bear in mind that `adjusted_r2` will need the arity `p`, which is not compatible with all other metrics; I can make it compatible by `partial` it with `p` for current `X_train`
      
* Add that tabular NN representative to the mix. At the moment, the most likely candidate is the ResNet mentioned in Gael et al.'s benchmark; I have the link in another notes file (` resuming-work-6Sep24.md` in `rbfmodels`), but let us reproduce it here:
    + https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/README.md#resnet-
    + paper is at https://arxiv.org/pdf/2106.11959: Revisiting Deep Learning Models for Tabular Data. 
    + Code at https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/README.md


* In addition to simple linear and regularised linear, could add robust linear (Huber) and basis functions linear (BSplines). See below.

* Expand the list of estimators; perhaps not need to report on all of them (for example, a representative of boosted trees might be enough, but no harm in doing CatBoost/LightGBM/XGBoost):
    - Simple Linear Regression
    - Regularised Linear Regression
    - Huber Regression (Robust)
    - BSplines/Cheb Regression
    - Symbolic Regression
    - Boosted Trees
        + XGBoost
        + sklearn's HistGradientBoostedRegressor (alternatively, LightGBM)
        + CatBoost
    - Random Forests
    - A DNN Tabular Regressor

* For the above, specially BSplines and **Symbolic Regression**, need to think of a **reasonable hyper-parameter space** across many datasets.

* Add Median Absolute Deviation (MedAD) to my list of metrics. So we are talking about Adjusted R^2, MAE, MedAD, RMSE.

* Instead of new repo move things into directories such as jun22 and sep24.

* **Optimise all learners on the same metric** (e.g., MSE); there is an issue here, which I mentioned under PolyRegr. Some of the polynomial regressors, as well as our ERBFN, throw some large residual at time, which can cause overflow when squaring/exponiating

* I’d need to keep train and test split (80/20 is fine) for my benchmark; there might be some post-fit analyses with train and held out data and I wouldn’t like tuning to have seen the whole data. 

* Obtain **inference time** as well.

* When running experiments, take a snapshot of Python installation (at least versions of key packages)
    - There is a ChatGPT inspired script somewhere...
    
* SHAP sensitivity analysis on ChebHuber on the AMES example (e.g., `cheb-huber-optuna-hs-ames-12Sep24`) looked promising in terms of XAI. Try also on ERBFN soon. 

* Make our ERBFN as compatible as possible with sklearn pipelines - might need to implement get features out, among others.
