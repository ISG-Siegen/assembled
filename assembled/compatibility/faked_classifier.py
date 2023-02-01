import time
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array, _check_y
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

from assembled.utils.logger import get_logger
from pandas.util import hash_array

logger = get_logger(__file__)


class FakedClassifier(BaseEstimator, ClassifierMixin):
    """A fake classifier that simulates a real classifier from the prediction data of the real classifier.

        We assume the input passed to init is the same format as training X (will be validated in predict).
        We store the prediction data with an index whereby the index is the hash of an instance.
            Some Assumptions of this:
                TODO: add checks for this
                    - Simulated Models return the same results for the same input instance
                    - Input data is only numeric (as the default preprocessor makes sure)


        !Warnings!:
            - If the simulated model returns different results for the same input instance (e.g., as a result of
            using cross-validation to produce the validation data), we set the prediction values for all duplicates
            to the first value seen for the duplicates.

        A Remark on Hashing if Duplicates are Present:
            The hash we are using is consistent between runs of the same INTERPRETER, i.e., Python Version
            (see https://stackoverflow.com/a/64356731). It is only consistent because we are hashing a tuple of
            numeric values. This would not work for hashes of strings without changing the code
            (see https://stackoverflow.com/a/2511075).

            Consistency is required if duplicates are present, because otherwise, in the current implementation,
            the prediction value selected to represent all duplicates might change and thus the data that is passed
            to an ensemble method would change.

            TODO: re-implement this, change hash method, or think of different approach to
                non-restrictive index management

    Parameters
    ----------
    simulate_time : bool, default=False'
        Whether the fake mode should also fake the time it takes to fit and predict.
        Note: currently we are not compensating for the overhead of the simulation in anyway or form.
              TODO: this is future work; does not support validation data....

    Parameters that are Simulated Attributes
    ----------
    oracle_X : array-like, shape (n_samples, n_features)
        The test input samples.
    predictions_ : array-like, shape (n_samples,)
        The predictions on the test input samples.
    oracle_index_:
        The predictions/confidence index list. Represent hash values for each instance of the simulation data to
        re-associated predict/predict_proba calls with the original predictions no matter the subset of the simulation
        data.
    simulate_n_features_in_: int, default=None
        The number of features seen during and used for validation.
    fit_time_ : float, default=0
        Time in seconds needed to fit the original real classifier.
    predict_time_ : float, default=0
            Time the real model took to evaluate/infer the predictions
    confidences_ : ndarray, shape (n_samples, n_classes)
        The confidences on the test input samples. We expect the confidences to be in the order of the classes as
        they appear in the training ground truth. (E.g.: by np.unique(y) or by a label encoder)
    predict_proba_time_ : float, default=0
        Time the real model took to evaluate/infer the confidences
    label_encoder : bool, default=false
        Whether we need to apply encoding to the predictions.
    model_metadata: Optional[dict], default=None
        Additional metadata for the model.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    le_ : LabelEncoder, object
        The label encoder created at :meth:`fit`.
    n_features_in_: int
        The number of features seen during fit.
    """

    def __init__(self, oracle_X=None, predictions_=None, confidences_=None, oracle_index_=None,
                 simulate_n_features_in_=None,
                 predict_time_: float = 0, predict_proba_time_: float = 0, fit_time_: float = 0,
                 simulate_time: bool = False, label_encoder=False, model_metadata=None):
        self.simulate_time = simulate_time
        self.label_encoder = label_encoder

        self.fit_time_ = fit_time_
        self.model_metadata = model_metadata

        # --- Init Parameter Validation
        if (oracle_X is not None) or (simulate_n_features_in_ is None):
            if confidences_ is None:
                oracle_X, predictions_ = self._validate_simulation_data(oracle_X, predictions_, reset=True)
            elif predictions_ is None:
                oracle_X, confidences_ = self._validate_simulation_data(oracle_X, confidences=confidences_, reset=True)
            else:
                # Neither is None
                oracle_X, predictions_, confidences_ = self._validate_simulation_data(oracle_X, predictions_,
                                                                                      confidences_, reset=True)
        else:
            self.simulate_n_features_in_ = simulate_n_features_in_

        self.predictions_ = predictions_
        self.confidences_ = confidences_

        if oracle_index_ is None:
            self.oracle_index_ = np.apply_along_axis(self._generate_index_from_row, 1, oracle_X)
        else:
            self.oracle_index_ = oracle_index_

        self.predict_time_ = predict_time_
        self.predict_proba_time_ = predict_proba_time_

        # Set to None for cloning etc
        self.oracle_X = None

    def fit(self, X, y):
        """Fitting the fake classifier, that is, doing nothing.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns
        """
        # --- Input Validation
        X, y = self._validate_data(X, y)

        # Store the classes seen during fit
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_

        if self.simulate_time:
            check_is_fitted(self, ["fit_time_"])
            time.sleep(self.fit_time_)

        return self

    def predict(self, X):
        """ Predicting with the fake classifier, that is, returning the previously stored predictions.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Vector containing the class labels for each sample.
        """

        # -- Check if fitted
        check_is_fitted(self, ["classes_", "n_features_in_", "predictions_", "oracle_index_",
                               "simulate_n_features_in_", "predict_time_", "le_"])
        if self.predictions_ is None:
            raise ValueError("Simulation of FakedClassifier is not fitted for predict.")

        # -- Input validation
        X = self._validate_data(X, reset=False)
        X = self._validate_simulation_data(X, reset=False)

        # -- Get Predictions from stored data
        # Get indices of data
        test_data_hash_indices = np.apply_along_axis(self._generate_index_from_row, 1, X)

        # Following Idea/Code From: https://www.statology.org/numpy-find-index-of-value/ to find index
        #   This will set the values of duplicates to the first occurrences of the duplicate
        #       (because that is the first index that is found by argsort).
        sorter = np.argsort(self.oracle_index_)
        prediction_indices = sorter[np.searchsorted(self.oracle_index_,
                                                    test_data_hash_indices, sorter=sorter)]

        # Update labels similar to that in the meta model
        if self.label_encoder:
            predictions = self.le_.transform(self.predictions_[prediction_indices])
        else:
            predictions = self.predictions_[prediction_indices]

        if self.simulate_time:
            time.sleep(self.predict_time_)

        return predictions

    def predict_proba(self, X):
        """ Predicting with the fake classifier, that is, returning the previously stored confidences.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples, n_classes)
            Returns the probability of each sample for each class in the model, where classes are ordered as they
            are in self.classes_.
        """

        # Validate length of confidences
        if len(self.classes_) != self.confidences_.shape[1]:
            raise ValueError("Confidences have a wrong shape of {}. Wrong number of enough classes.".format(
                self.confidences_.shape))

        # -- Check if fitted
        check_is_fitted(self, ["classes_", "n_features_in_", "confidences_", "oracle_index_",
                               "simulate_n_features_in_", "predict_proba_time_"])
        if self.confidences_ is None:
            raise ValueError("Simulation of FakedClassifier is not fitted for predict_proba.")

        # -- Input validation
        X = self._validate_data(X, reset=False)
        X = self._validate_simulation_data(X, reset=False)

        # -- Get Predictions from stored data
        # Get indices of data
        test_data_hash_indices = np.apply_along_axis(self._generate_index_from_row, 1, X)

        # Following Idea/Code From: https://www.statology.org/numpy-find-index-of-value/ to find index
        #   This will set the values of duplicates to the first occurrences of the duplicate
        #       (because that is the first index that is found by argsort).
        sorter = np.argsort(self.oracle_index_)
        confidences_indices = sorter[np.searchsorted(self.oracle_index_,
                                                     test_data_hash_indices, sorter=sorter)]
        confidences = self.confidences_[confidences_indices]

        if self.simulate_time:
            time.sleep(self.predict_proba_time_)

        return confidences

    def _validate_simulation_data(self, X, predictions="no_validation", confidences="no_validation",
                                  reset=True):
        """Validate simulation data and set or check the `simulate_n_features_in_` and `simulate_classes_` attribute.

        Parameters
        ----------
        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features)
            The input samples.
        predictions : array-like of shape (n_samples,), default='no_validation'
            The predictions data used for simulation.
        reset : bool, default=True
            Whether to reset the `simulate_n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.

        Returns
        -------
        out : {ndarray, sparse matrix} or tuple of these
            The validated input. A tuple is returned if both `X` and `predictions` are
            validated.
        """
        # --- Check X
        X = check_array(X)

        # Check feature
        if reset:
            self.simulate_n_features_in_ = X.shape[1]
        else:
            if X.shape[1] != self.simulate_n_features_in_:
                raise ValueError("X has not enough features.")

        # --- Check Predictions
        if not (isinstance(predictions, str) and predictions == "no_validation"):
            predictions = _check_y(predictions)

            # Check equality of simulation data
            if predictions.shape[0] != X.shape[0]:
                raise ValueError("Shape mismatch for X and predictions: X has {}, predictions has {}".format(
                    X.shape, predictions.shape))

        # --- Check Confidences
        if not (isinstance(confidences, str) and confidences == "no_validation"):
            if isinstance(confidences, pd.DataFrame):
                confidences = confidences.to_numpy()  # remove sparse data using to_numpy
            confidences = check_array(confidences, accept_sparse=False)

            # Check equality of simulation data
            if confidences.shape[0] != X.shape[0]:
                raise ValueError("Shape mismatch for X and confidences: X has {}, confidences has {}".format(
                    X.shape, confidences.shape))

        # --- Determine return
        res = [val for val in [X, predictions, confidences]
               if not (isinstance(val, str) and val == "no_validation")]

        # Fail save to avoid list-like return
        if len(res) == 1:
            return res[0]

        return tuple(res)

    @staticmethod
    def _generate_index_from_row(x):
        # See Class docu above why this works
        return hash(tuple(x))


# -- Additional Functions for FakedClassifiers Usage
def _initialize_fake_models(test_base_model_train_X, test_base_model_train_y, val_base_model_train_X,
                            val_base_model_train_y, known_X, known_predictions, known_confidences, pre_fit_base_models,
                            base_models_with_names, label_encoder, to_confidence_name, predictor_descriptions):
    # !WARNING!: Long text to explain to myself why the following works
    # Theoretically, we would need to build two separate base model sets for validation and test, because the base
    #   models might have been fitted differently for the validation predictions and test predictions.
    #   That is, in the case of a refit.
    # However, in our "lookup-table-like" fake base model, the fit method validates that the number of features
    #   is correct. Which it must be for both cases in our setup. Otherwise, it would be a bug.
    # Moreover, in classification it uses np.unique(y) to find the classes used to convert the predictions. This is,
    #   however, almost always the same for val_y and test_y because we are always doing a stratified splits.

    # TODO: fix this problem below
    #   If refit + the validation data, with enough classes, it can happen that test_base_model_train_y has a different
    #   number of classes than val_base_model_train_y, then a problem would exist. This is not yet supported / fixed.

    # In any other case, the difference between data used to fit the base model for validation or test (outer
    #   evaluation) can be ignored. The base model can be (pre-)fitted on either validation or test data.
    # It is only important that known_X contains all instances, known_predictions/confidences all output values
    #   that would be needed during prediction for validation (ensemble fit) or outer evaluation (ensemble predict),
    #   and the label encoder all possible classes.

    # -- Set values to one of validation or training data here
    classes_ = np.unique(test_base_model_train_y)
    X_train = test_base_model_train_X
    y_train = test_base_model_train_y

    # - Sanity Check to make sure we are not in an ill-defined split and everything described as above holds
    if list(classes_) != list(np.unique(val_base_model_train_y)):
        val_classes = np.unique(val_base_model_train_y)
        logger.info("Classes found in Validation and Test data used to fit the base model are not identical."
                    + "\nClasses in test data: {}".format(list(classes_))
                    + "\nClasses in val data: {}".format(list(val_classes)))
        raise NotImplementedError("This can only occur when refit is true. This is not yet supported.")

    # -- Build fake base models
    faked_base_models = []
    add_metadata = predictor_descriptions is not None

    for model_name in list(known_predictions):
        # Sort Predictions to the default predict_proba style and get predictions + confidences for a specific model
        model_confidences = known_confidences[[to_confidence_name(model_name, class_name) for class_name in classes_]]
        model_predictions = known_predictions[model_name]

        # Fit the FakedClassifier
        fc = FakedClassifier(known_X, model_predictions, model_confidences, label_encoder=label_encoder,
                             model_metadata=predictor_descriptions[model_name] if add_metadata else None)

        # -- Set fitted or not (sklearn vs. deslib)
        if pre_fit_base_models:
            fc.fit(X_train, y_train)

        # -- Set result output (sklearn vs. deslib)
        res = fc
        if base_models_with_names:
            res = (model_name, res)

        faked_base_models.append(res)

    return faked_base_models


def probability_calibration_for_faked_models(base_models, X_meta_train, y_meta_train, probability_calibration_type,
                                             pre_fit_base_models):
    # -- Simply return base models without changes
    if probability_calibration_type == "no":
        return base_models

    # -- Select calibration method
    if probability_calibration_type != "auto":
        # We assume the input has been validated and only "sigmoid", "isotonic" are possible options
        cal_method = probability_calibration_type
    else:
        # TODO-FUTURE: perhaps add something that selects method based on base model types?

        # Select method based on number of instances
        cal_method = "isotonic" if len(y_meta_train) > 1100 else "sigmoid"

    # --Build calibrated base models
    cal_base_models = []
    for bm_data in base_models:

        # - Select the base model
        if isinstance(bm_data, tuple):
            bm = bm_data[1]
        else:
            bm = bm_data

        # - Determine how to process the base models
        if pre_fit_base_models:
            cal_bm = CalibratedClassifierCV(bm, method=cal_method, cv="prefit").fit(X_meta_train, y_meta_train)
        else:
            # With cv=2 we have less overhead with fake base models.
            # Once our current fake base model structure changes, we need to change this as well.
            cal_bm = CalibratedClassifierCV(bm, method=cal_method, ensemble="False", cv=2)

        # - Set base model
        if isinstance(bm_data, tuple):
            cal_base_models.append((bm_data[0], cal_bm))
        else:
            cal_base_models.append(cal_bm)

    return cal_base_models
