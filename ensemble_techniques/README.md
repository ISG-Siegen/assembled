# Assembled's Supported Ensemble Techniques

It is important to note, that we utilize ensemble techniques of existing libraries and new/custom techniques through
simulating base models based on our data. To execute ensemble techniques on our benchmark, we have created so-called "
FakedClassifiers". These represent base models (sklearn estimators) to a library like sklearn or DESlib and thus allows
us to use these natively. Employing these fake base models is the default case for our benchmark and all our code. We
have also added a wrapper to use methods that only work with the predictions (like autosklearn's ensemble selection).

The fake base models are created on-the-fly when you run the "run_ensemble_on_all_folds" function of a Metatask and then
passed to the ensemble technique (we assume that a list of base models or something similar is the first argument of the
technique's initialization).

Please consult the first version of Assembled-OpenML (see the other branches) to see an alternative on how to use/create
a benchmark of ensemble techniques that works with the data directly instead of base models.

## Overview

This directory contains all supported ensemble techniques as well as code to enable their usage. The code for the
FakedClassifiers and other compatability utilities can be found in the `assembled` directory.

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
* Stacking and Voting are both implemented to fit the base models themselves. We can accommodate for this with a few
  workarounds for our faked base models. However, Stacking also wants to employ cross_val_predict on the base models. We
  can not support this for the faked base models, as the data from OpenML and their theoretical concept both do not
  support this. Hence, we created our own versions of stacking and voting where we can pass already fitted base models.
  Furthermore, for stacking we added the possibility to pass data such that it is only used to train final_estimator.
    * Technically ,we can support the usage of cross_val_preds as is with our faked base models. This would result in
      blending behavior without chaining the code of stacking. This however becomes very problematic once you want to
      add things like base model calibration.

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

* Please be aware, adding calibration to the base models increases the overhead for fit and predict. Especially if the
  ensemble techniques can not work with pre-fitted models. Consider the following, adding calibration is like adding
  another wrapper model for each base model (on-the-fly or during pre-processing).
