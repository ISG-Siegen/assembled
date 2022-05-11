# Assembled's Supported Ensemble Techniques

This directory contains all supported ensemble techniques as well as code to enable their usage. The code for the
FakedClassifiers and other compatability utilities can be found in the `assembledopenml` directory.

The file `collect_ensemble_techniques.py` contains the code to invoke the ensemble technique and the parameters passed
to the ensemble technique at run time. Moreover, we include additional parameters required by
the `run_ensemble_on_all_folds` function of a metatask to know how to execute and handle these ensemble techniques.

We support the following Ensemble Techniques through FakedClassifiers:

| Library | Techniques | n_techniques | Comment |
|---|---|---|---|
| scikit-learn | Stacking/Blending, Voting | 2 | - |
| DESlib | Many Dynamic (Ensemble/Classifier) Selection Techniques | 19 | We are not using static techniques.  |
| auto-sklearn | Ensemble Selection | 1 | - |
| Custom | VirtualBest, SingleBest, Dynamic Selection with EPMs | 3 | - |

## General Remarks

* For reproducibility, we guarantee that each technique is initialized with a seed-based random state for each
  evaluation run (once for all folds of a task for each technique).

## Known Issues

### Sickit-learn

* We can not employ multiprocessing (`n_jobs=-1`) to stacking or voting. Our best guess is, that the FakedClassifiers
  somehow break the multiprocessing implementation, because multiprocessing is used to train and evaluate the base
  models. So far, we have not found the reason for this or were able to fix it.
    * This is, however, not really relevant as the cost of multiprocessing is higher than fitting/predicting in
      sequential fashion for the FakedClassifiers. Moreover, we can enable multiprocessing for the final estimator by
      hand.
* Stacking and Voting are usually fit on the original training data. However, we can fit them on our meta-train split,
  because the both only require the original training data to fit the base models. But the base models' fit function
  does nothing besides storing some structural information which we can also obtain from the meta-train data. As a
  result the following behavior is induced:
    * Stacking is not trained on the cross_val_predictions from the full training data but simply on a meta-train split.
      With the data we have and how we use fake model, this is the default behavior if we pass the base models to
      stacking. In other words, the base models can work with the fact that stacking calls cross_val_predictions on
      them, because they are not re-fitted (fit does not do anything for the fake models) and they return only the
      meta-train predictions.
    * Voting is not trained at all (only the base models). For the sake of our usage here, we could fit voting on the
      original data. But to support calibration we had to remove it and treat it like Stacking. That is, we fit on the
      meta-train split.
        * If we used calibration and fit on the original train data, the CalibratedClassifierCV would try to get the
          prediction data as if the base model had been trained on (potentially new) folds. This is not yet supported by
          the faked base models because/and this data does not exist on OpenML. The same would happen if we try to fit
          stacking on the original data.

### Autosklearn

* The Bagging parameter is not supported for ensemble selection. We have not enabled it so far.
* Depending on the used metric, ensemble selection can become very inefficient. Since we are using AUROC most often,
  ensemble selection is very inefficient.

### DESlib

* DESlib does not support all metrics of sklearn. Moreover, it does not support all metrics consistently. We can not
  pass a metric object/callable but have to pass a string. Yet, sklearn's `roc_auc_score` string is not supported.
  Hence, we had to use another metric for techniques that require it (like DESClustering). In such cases, we used
  `accruacy_score`.
* Generally, it is important to note that most Dynamic Selection techniques in DESlib are implicitly optimizing the
  accuracy. Local techniques can most often only use accuracy by design and alternatives are not supported.
    * Keep this in mind for any evaluation!
* Most often, we are using the default meta-model for DESlib methods. You could also pass different methods. We had to
  do this for DESClustering, because otherwise the algorithm tried to initialize KMeans with an n_jobs parameter.
  However, we have found not sklearn version that supports an n_jobs parameter for KMeans.
    * It is important to note, that DESlib is always using multiprocessing with `n_jobs=-1` by default!

### Custom

* We are not using the static techniques of DESlib to get the VirtualBest and SingleBest. The VirtualBest did run into
  an error and the SingleBest is not optimal for confidences. Hence, to have a higher control, we added our own code to
  get the SingleBest and VirtualBest.

### Probability Calibration

* Please be aware, adding calibration to the base models increases the overhead for fit and predict. Especially if
  the ensemble techniques can not work with pre-fitted models. 
  Consider the following, adding calibration is like adding another model for each base model (on-the-fly or during
  pre-processing). 