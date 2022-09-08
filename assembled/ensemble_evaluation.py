import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from typing import Optional, Callable, List

from assembled.metatask import MetaTask
from assembled.utils.logger import get_logger
from assembled.utils.isolation import isolate_function
from assembled.utils.preprocessing import check_fold_data_for_ensemble
from assembled.utils.data_mgmt import save_results
from assembled.compatibility.faked_classifier import probability_calibration_for_faked_models, _initialize_fake_models

logger = get_logger(__file__)


def evaluate_ensemble_on_metatask(metatask: MetaTask, technique, technique_args: dict, technique_name,
                                  folds_to_run: Optional[List[int]] = None,
                                  use_validation_data_to_train_ensemble_techniques: bool = False,
                                  meta_train_test_split_fraction: float = 0.5, meta_train_test_split_random_state=0,
                                  pre_fit_base_models: bool = False, base_models_with_names: bool = False,
                                  label_encoder=False, preprocessor=None, output_dir_path=None,
                                  store_results: str = "sequential", save_evaluation_metadata: bool = False,
                                  oracle=False, probability_calibration="no", predict_method: str = "predict",
                                  return_scores: Optional[Callable] = None, verbose: bool = False,
                                  isolate_ensemble_execution: bool = False, refit: bool = False):
    """Run an ensemble technique on all folds and return the results

    The current implementation builds fake base models by default such that we can evaluate methods this way.

    Parameters
    ----------
    metatask: Metatask
        The metatask on which we want to evaluate the ensemble.
    technique: sklearn estimator like object
        The Ensemble Technique (as a sklearn like fit/predict style) model.
    technique_args: dict
        The arguments that shall be supplied to the ensemble
    technique_name: str
        Name of the technique used to identify the technique later on in the saved results.
    folds_to_run: List of integers, default=None
        If None, we run the ensemble on all folds.
        Otherwise, we run the ensemble on all specified folds.
    use_validation_data_to_train_ensemble_techniques: bool, default=False
        Whether to use validation data to train the ensemble techniques (through the faked base models) and use the
        fold's prediction data to evaluate the ensemble technique.
        If True, the metataks requires validation data. If False, test predictions of a fold are split into
        meta_train and meta_test subsets where the meta_train subset is used to train the ensemble techniques and
        meta_test to evaluate the techniques.
    meta_train_test_split_fraction: float, default=0.5
        The fraction for the meta train/test split. Only used if
    meta_train_test_split_random_state: int, default=0
        The randomness for the meta train/test split.
    pre_fit_base_models: bool, default=False
        Whether or not the base models need to be fitted to be passed to the ensemble technique.
    base_models_with_names: bool, default=False
        Whether or not the base models' list should contain the model and its name.
    label_encoder: bool, default=False
        Whether the ensemble technique expects that a label encoder is applied to the fake models. Often required
        for sklearn ensemble techniques.
    preprocessor: sklearn-like transformer, default=None
        Function used to preprocess the data for later. called fit_transform on X_train and transform on X_test.
        If None, the default preprocessor is ued. The default preprocessor encodes categories as ordinal numbers
        and fills missing values.
    output_dir_path: str, default=None
        Path to directory where the results of the folds shall be stored. If none, we do not store anything.
        We assume existing files are in the correct format for store_results="sequential" and will create
        a new file if it does not exit.
        Here, no option to purge/delete existing files is given. This is must be done in an outer scope.
    store_results: {"sequential", "parallel"}, default="sequential"
        How to store the results of a fold's evaluation.
            - "sequential": Store the results of all evaluated methods in one unique file.
            - "parallel": Store each fold's results in different files such that they can later be merged easily.
    save_evaluation_metadata: bool, default=False
        If true, save evaluation metadata in a (separate) file. Otherwise, do not save run metadata.
        Currently evaluation metadata include: {"fit_time", "predict_time"}
        The metadata is stored for each fold.
    oracle: bool, default=False
        Whether the ensemble technique is an oracle. If true, we pass and call the method differently.
    probability_calibration: {"sigmoid", "isotonic", "auto", "no"}, default="no"
        What type of probability calibration (see https://scikit-learn.org/stable/modules/calibration.html)
        shall be applied to the base models:

            - "sigmoid": Use CalibratedClassifierCV with method="sigmoid"
            - "isotonic": Use CalibratedClassifierCV with method="isotonic"
            - "auto": Determine which method to use for CalibratedClassifierCV depending on the number of instances.
            - "no": Do not use probability calibration.

        If pre_fit_base_models is False, CalibratedClassifierCV is employed with ensemble="False" to simulate
        cross_val_predictions by our Faked Base Models.
        If pre_fit_base_models is True, CalibratedClassifierCV is employed with cv="prefit" beforehand such that
        we "replace" the base models with calibrated base models.
    predict_method: str in {"predict", "predict_proba"}, default="predict"
        Predict method used. For regression only "predict" works. For classification, predict_proba will return
        prediction probabilities (a.k.a. confidences).
    return_scores: Callable, default=None
        If the evaluation shall return the scores for each fold. If not None, a metric function is expected.
    verbose: bool, default=False
        If True, evaluation status information are logged.
    isolate_ensemble_execution: bool, default=False
        If True, we isolate the execution of the ensemble in its own subprocess. This avoids problems
        with memory leakage or other problems from implementations of the ensemble.
        !WARNING! Only works on Linux currently; FIXME: Either a bug or not possible on Windows, dont know.
    refit: bool, default=False
        Set to true if the base models have been re-fitted before predicting for test data.
    """
    # TODO -- Parameter Preprocessing / Checking
    #   Add safety check for file path here or something
    #   Check if probability_calibration has correct string names
    #   Check metric / scorer object

    if use_validation_data_to_train_ensemble_techniques and (not metatask.use_validation_data):
        raise ValueError("Metatask has no validation data but use_validation_to_train_ensemble_techniques is True.")

    if isolate_ensemble_execution:
        import platform
        if platform.system() == "Windows":
            raise ValueError("The option isolate_ensemble_execution can currently not be used on Windows!")

    if store_results not in ["sequential", "parallel"]:
        raise ValueError("Store_results parameter has a wrong value. Allowed are: {}. Got: {}".format(
            ["sequential", "parallel"], store_results))

    if predict_method not in ["predict", "predict_proba"]:
        raise ValueError("predict_method parameter has a wrong value. Allowed are: {}. Got: {}".format(
            ["predict", "predict_proba"], predict_method))

    # -- Get Arguments for different steps of the ensemble evaluation
    val_data = use_validation_data_to_train_ensemble_techniques
    if val_data:
        val_data_kwargs = {"verbose": verbose, "val_data": val_data, "refit": refit}
    else:
        val_data_kwargs = {"verbose": verbose, "val_data": val_data,
                           "meta_train_test_split_fraction": meta_train_test_split_fraction,
                           "meta_train_test_split_random_state": meta_train_test_split_random_state}

    base_model_build_kwargs = {"pre_fit_base_models": pre_fit_base_models,
                               "base_models_with_names": base_models_with_names,
                               "label_encoder": label_encoder, "probability_calibration": probability_calibration,
                               "to_confidence_name": metatask.to_confidence_name, "verbose": verbose}

    ensemble_run_kwargs = {"oracle": oracle, "predict_method": predict_method, "verbose": verbose,
                           "isolate_ensemble_execution": isolate_ensemble_execution}

    # -- Iterate over Folds
    fold_scores = []
    for fold_idx, X_train, X_test, y_train, y_test, val_base_predictions, test_base_predictions, \
        val_base_confidences, test_base_confidences in metatask.yield_evaluation_data(folds_to_run):

        # --- Add fold specific information to kwargs
        if val_data:
            # -- Get iloc indices to identify what part of the prediction data is used for validation
            train_X_indices = X_train.index.to_numpy()
            # Get iloc indices of validation data (Default value is for backwards compatibility)
            fold_validation_indices = metatask.validation_indices.get(fold_idx, train_X_indices)
            # iloc_indices corresponds to np.arange(len(X_train)) for cross-validation
            val_data_kwargs["iloc_indices"] = np.intersect1d(fold_validation_indices, train_X_indices,
                                                             return_indices=True)[2]

            # -- Get names needed to combine test and val predictions later
            val_data_kwargs["p_rename"] = {metatask.to_validation_predictor_name(col_name, fold_idx): col_name for
                                           col_name in list(test_base_predictions)}
            val_data_kwargs["c_rename"] = {metatask.to_validation_predictor_name(col_name, fold_idx): col_name for
                                           col_name in list(test_base_confidences)}

        if verbose:
            logger.info("Start Evaluation for Fold {}/{}...".format(fold_idx + 1, metatask.max_fold + 1))

        ensemble_test_y, y_pred_ensemble_model, run_meta_data, test_indices = \
            _run(X_train, X_test, y_train, y_test, val_base_predictions, val_base_confidences,
                 test_base_predictions, test_base_confidences, preprocessor,
                 base_model_build_kwargs, val_data_kwargs,
                 technique, technique_args, ensemble_run_kwargs)

        # -- Add evaluation settings metadata
        if save_evaluation_metadata:
            run_meta_data["ensemble_model_arguments"] = str(technique_args)
            run_meta_data["validation_data_used"] = use_validation_data_to_train_ensemble_techniques
            run_meta_data["meta_train_test_split_fraction"] = meta_train_test_split_fraction
            run_meta_data["meta_train_test_split_random_state"] = meta_train_test_split_random_state
            run_meta_data["pre_fit_base_models"] = pre_fit_base_models
            run_meta_data["base_models_with_names"] = base_models_with_names
            run_meta_data["label_encoder"] = label_encoder
            run_meta_data["preprocessor"] = "No Preprocessor" if preprocessor is None else str(preprocessor).replace(
                " ", "").replace("\n", "")
            run_meta_data["oracle"] = oracle
            run_meta_data["probability_calibration"] = probability_calibration

        # -- Post Process Results
        save_results(output_dir_path, store_results, save_evaluation_metadata, ensemble_test_y, y_pred_ensemble_model,
                     fold_idx, technique_name, test_indices, metatask.openml_task_id, run_meta_data, predict_method,
                     metatask.class_labels)

        # -- Save scores for return
        if return_scores is not None:
            fold_scores.append(return_scores(ensemble_test_y, y_pred_ensemble_model))

        if verbose:
            logger.info("Finished Evaluation.")

    # -- Return Score
    if return_scores is not None:
        return fold_scores


# -- Overall code
def _run(X_train, X_test, y_train, y_test, val_base_predictions, val_base_confidences,
         test_base_predictions, test_base_confidences, preprocessor, base_model_build_kwargs, val_data_kwargs,
         technique, technique_args, ensemble_run_kwargs):
    """Wrappers like _run() or _get_evaluation_input are used/needed to avoid memory leakage."""
    # Get everything needed for fit and predict with the ensemble
    ensemble_train_X, ensemble_test_X, ensemble_train_y, ensemble_test_y, test_indices, base_models = \
        _get_evaluation_input(X_train, X_test, y_train, y_test, val_base_predictions, val_base_confidences,
                              test_base_predictions, test_base_confidences, preprocessor,
                              val_data_kwargs, base_model_build_kwargs)

    y_pred_ensemble_model, run_meta_data = _run_ensemble_wrapper(base_models, technique, technique_args,
                                                                 ensemble_train_X, ensemble_test_X, ensemble_train_y,
                                                                 ensemble_test_y, **ensemble_run_kwargs)

    return ensemble_test_y, y_pred_ensemble_model, run_meta_data, test_indices


def _get_evaluation_input(X_train, X_test, y_train, y_test, val_base_predictions,
                          val_base_confidences, test_base_predictions, test_base_confidences,
                          preprocessor, val_data_kwargs, base_model_build_kwargs):
    # -- Employ Preprocessing and input validation
    test_X_indices = X_test.index.tolist()
    X_train, X_test, y_train, y_test = check_fold_data_for_ensemble(X_train, X_test, y_train, y_test,
                                                                    preprocessor)

    # - Validation Data
    val_data = val_data_kwargs["val_data"]
    val_kwargs = val_data_kwargs.copy()
    del val_kwargs["val_data"]

    if not val_data:
        ensemble_train_X, ensemble_test_X, ensemble_train_y, ensemble_test_y, test_indices, base_models = \
            _get_data_for_ensemble_and_base_models_without_validation(X_train, X_test, y_train, y_test,
                                                                      test_base_predictions, test_base_confidences,
                                                                      test_X_indices, base_model_build_kwargs,
                                                                      **val_kwargs)
    else:
        # All elements of the test data are test indices if we have validation data
        test_indices = test_X_indices

        ensemble_train_X, ensemble_test_X, ensemble_train_y, ensemble_test_y, base_models = \
            _get_data_for_ensemble_and_base_models_with_validation(X_train, X_test, y_train, y_test,
                                                                   test_base_predictions, test_base_confidences,
                                                                   val_base_predictions, val_base_confidences,
                                                                   base_model_build_kwargs,
                                                                   **val_kwargs)

    return ensemble_train_X, ensemble_test_X, ensemble_train_y, ensemble_test_y, test_indices, base_models


# -- Data for Ensemble and Base Models Getters
def _get_data_for_ensemble_and_base_models_without_validation(X_train, X_test, y_train, y_test, test_base_predictions,
                                                              test_base_confidences, test_X_indices,
                                                              base_model_build_kwargs, meta_train_test_split_fraction,
                                                              meta_train_test_split_random_state, verbose):
    """Split for ensemble technique evaluation only on fold predictions to get trainings data for the ensemble

    Input to _build_fake_base_models Explanation
    -------
    val_base_model_train_X = test_base_model_train_X = X_train,
    val_base_model_train_y = test_base_model_train_y = y_train:
        In this case, the base models have been fitted on the same data for the validation and test data,
        because we take a subset of the test data as validation data.

    ensemble_train_X, ensemble_train_y:
        The split of existing validation data is used.

    fake_base_model_known_X = X_test
    fake_base_model_known_predictions = test_base_predictions
    fake_base_model_known_confidences = test_base_confidences
        The base models only know X_test and the predictions existing for X_test.
        All prediction data exists only for X_test in this case.

    """
    if verbose:
        logger.info("Getting validation data by splitting the test data.")
    # Split of the original test data corresponding to the parts of the predictions used for train and test
    ensemble_train_X, ensemble_test_X, ensemble_train_y, ensemble_test_y, _, test_indices \
        = train_test_split(X_test, y_test, test_X_indices, test_size=meta_train_test_split_fraction,
                           random_state=meta_train_test_split_random_state, stratify=y_test)

    base_models = _build_fake_base_models(X_train, y_train, X_train, y_train, ensemble_train_X, ensemble_train_y,
                                          X_test, test_base_predictions, test_base_confidences,
                                          **base_model_build_kwargs)

    return ensemble_train_X, ensemble_test_X, ensemble_train_y, ensemble_test_y, test_indices, base_models


def _get_data_for_ensemble_and_base_models_with_validation(X_train, X_test, y_train, y_test,
                                                           test_base_predictions, test_base_confidences,
                                                           val_base_predictions, val_base_confidences,
                                                           base_model_build_kwargs, refit, iloc_indices, p_rename,
                                                           c_rename, verbose):
    """Build data for validation based evaluation

    Fake Model must be able to predict for all instances of the validation data and the fold's predictions.
    Hence, it must get all the data needed for that.

    Input to _build_fake_base_models Explanation
    -------
    test_base_model_train_X, val_base_model_train_X, test_base_model_train_y, val_base_model_train_y
        Depend on combination of refit and validation strategy. See, if-else case below fore details.

    ensemble_train_X, ensemble_train_y:
        Equals the instances of the validation data

    fake_base_model_known_X, fake_base_model_known_predictions, fake_base_model_known_confidences
        The base model knows all of X_test and its predictions.
            (X_test, test_base_predictions, test_base_confidences)
        Additionally it knows all validation predictions and corresponding instances.
            (X_train[iloc_indices], val_base_predictions.iloc[iloc_indices], val_base_confidences.iloc[iloc_indices])
        Hence, we need to stack/concat these data points and give it to the base model
    """
    if verbose:
        logger.info("Getting validation data from existing validation data.")

    # -- Determine which data points are not in the validation data
    # All iloc indices that are not in the validation data
    not_in_validation_data = np.setdiff1d(np.arange(len(X_train)), iloc_indices)
    # Determine if holdout validation or not
    holdout = False if not_in_validation_data.size == 0 else True

    # -- Get all data that the base models must know
    # We need to combine predictions / confidences (keep as DF for columns)
    # Have to rename columns to use DFs with validation data, here remove fold postfix to achieve this
    fake_base_model_known_predictions = pd.concat([val_base_predictions.iloc[iloc_indices].rename(columns=p_rename),
                                                   test_base_predictions], axis=0)
    fake_base_model_known_confidences = pd.concat([val_base_confidences.iloc[iloc_indices].rename(columns=c_rename),
                                                   test_base_confidences], axis=0)

    # -- Here, multiple things could be:
    if holdout and (not refit):
        # Only fitted for both on the same subset of the training data
        test_base_model_train_X = val_base_model_train_X = X_train[not_in_validation_data]
        test_base_model_train_y = val_base_model_train_y = y_train[not_in_validation_data]
    elif holdout and refit:
        # Fitted on two different sets
        val_base_model_train_X = X_train[not_in_validation_data]
        val_base_model_train_y = y_train[not_in_validation_data]
        test_base_model_train_X = X_train
        test_base_model_train_y = y_train
    else:
        # -- Case for: 3. Cross-val, refit and 4. cross-val, no refit
        # For our purposes, we just need to know that it was fitted on the whole data (at some point for cv)
        # For not refit this holds, because the CV models sees all training data.
        test_base_model_train_X = val_base_model_train_X = X_train
        test_base_model_train_y = val_base_model_train_y = y_train

    ensemble_train_X = X_train[iloc_indices]
    ensemble_train_y = y_train[iloc_indices]
    ensemble_test_X = X_test
    ensemble_test_y = y_test

    fake_base_model_known_X = np.vstack((X_train[iloc_indices], X_test))

    base_models = _build_fake_base_models(test_base_model_train_X, test_base_model_train_y,
                                          val_base_model_train_X, val_base_model_train_y,
                                          ensemble_train_X, ensemble_train_y,
                                          fake_base_model_known_X, fake_base_model_known_predictions,
                                          fake_base_model_known_confidences, **base_model_build_kwargs)

    return ensemble_train_X, ensemble_test_X, ensemble_train_y, ensemble_test_y, base_models


# -- Fake Base Model Code
def _build_fake_base_models(test_base_model_train_X, test_base_model_train_y,
                            val_base_model_train_X, val_base_model_train_y,
                            ensemble_train_X, ensemble_train_y, fake_base_model_known_X,
                            fake_base_model_known_predictions, fake_base_model_known_confidences,
                            pre_fit_base_models, base_models_with_names, label_encoder, probability_calibration,
                            to_confidence_name, verbose):
    """Build fake base models from data.

    New Parameters (rest can be found in signature of evaluate_ensemble_on_metatask)
    ----------
    test_base_model_train_X: array-like
        The instances on which the base models are trained for the (outer) evaluation
    test_base_model_train_y: array-like
        The labels on which the base models are trained for the (outer) evaluation
    val_base_model_train_X:
        The instances on which the base models are trained for the validation data.
    val_base_model_train_y: array-like
        The labels on which the base models are trained for the validation data.
    ensemble_train_X: array-like
        The instances on which the ensemble technique is to be trained.
    ensemble_train_y: array-like
        The labels on which the ensemble technique is to be trained.
    fake_base_model_known_X: array-like
        The instances from the test or training data for which the base model must know predictions.
    fake_base_model_known_predictions: pd.DataFrame
        Predictions for the instances of fake_base_model_known_X
    fake_base_model_known_confidences: pd.DataFrame
        Confidences for the instances of fake_base_model_known_X
    to_confidence_name: Callable
        Function that transforms predictor name and class name into confidence column name.

    # Ensemble train X,y: ensemble_train_X, ensemble_train_y
    # Ensemble test X,y: ensemble_test_X, ensemble_test_y
    # All instances on which the BMs have data: fake_base_model_known_X
    # All Predictions for all known instances: fake_base_model_known_predictions
    # All confidences for all known instances: fake_base_model_known_confidences
    # Base model training X,y: val_base_model_train_X, val_base_model_train_y (data trained on for validation pred)
    # Base model training X,y: test_base_model_train_X, test_base_model_train_y (data trained on for test pred)
    """
    if verbose:
        logger.info("Build Fake Base Models")

    # -- Build ensemble technique
    base_models = _initialize_fake_models(test_base_model_train_X, test_base_model_train_y,
                                          val_base_model_train_X, val_base_model_train_y, fake_base_model_known_X,
                                          fake_base_model_known_predictions, fake_base_model_known_confidences,
                                          pre_fit_base_models, base_models_with_names, label_encoder,
                                          to_confidence_name)

    # -- Probability Calibration
    base_models = probability_calibration_for_faked_models(base_models, ensemble_train_X, ensemble_train_y,
                                                           probability_calibration, pre_fit_base_models)

    return base_models


def _run_ensemble_wrapper(base_models, technique, technique_args, ensemble_train_X, ensemble_test_X, ensemble_train_y,
                          ensemble_test_y, oracle, predict_method, isolate_ensemble_execution, verbose):
    # -- Run Ensemble for Fold
    if verbose:
        logger.info("Run Ensemble on Fold")

    func_args = (base_models, technique, technique_args, ensemble_train_X, ensemble_test_X, ensemble_train_y,
                 ensemble_test_y, oracle, predict_method)
    func = _run_ensemble_on_data

    if isolate_ensemble_execution:
        y_pred_ensemble_model, run_meta_data = isolate_function(func, *func_args)
    else:
        y_pred_ensemble_model, run_meta_data = func(*func_args)

    return y_pred_ensemble_model, run_meta_data


def _run_ensemble_on_data(base_models, technique, technique_args, ensemble_train_X, ensemble_test_X,
                          ensemble_train_y, ensemble_test_y, oracle, predict_method):
    """Run an ensemble technique for the given data and return its predictions

    New Parameters (rest can be found in signature of run_ensemble_on_all_folds or _build_fake_base_models)
    ----------
    base_models: List of fake base models
        A list of base models that shall be used by the ensemble technique.
    ensemble_test_X: array-like
        The instances on which the ensemble technique shall predict on.
    ensemble_test_y: array-like
        The labels which the ensemble technique shall predict (only used for oracle-like techniques).
    """

    # -- Build ensemble model
    ensemble_model = technique(base_models, **technique_args)

    # -- Fit and Predict
    st = time.time()
    ensemble_model.fit(ensemble_train_X, ensemble_train_y)
    fit_time = time.time() - st

    st = time.time()
    if oracle:

        if predict_method == "predict_proba":
            raise NotImplemented("Prediction Probabilities are not supported (yet) for Oracle Predictors!")

        # Special case for virtual oracle-like predictors
        y_pred_ensemble_model = ensemble_model.oracle_predict(ensemble_test_X, ensemble_test_y)

    else:

        if predict_method == "predict":
            y_pred_ensemble_model = ensemble_model.predict(ensemble_test_X)
        else:
            y_pred_ensemble_model = ensemble_model.predict_proba(ensemble_test_X)

    predict_time = time.time() - st

    run_meta_data = {"fit_time": fit_time, "predict_time": predict_time}

    return y_pred_ensemble_model, run_meta_data
