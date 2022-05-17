import time
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted, check_array, _check_y
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV


class FakedClassifier(BaseEstimator, ClassifierMixin):
    """ A fake classifier that simulates a real classifier from the prediction data of the real classifier.

        We assume the input passed to init is the same format as training X (will be validated in predict).
        We store the prediction data with an index whereby the index is the hash of an instance.
            Some Assumptions of this:
                - Simulated Models return the same results for the same input instance

    Parameters
    ----------
    simulate_time : bool, default=False'
        Whether the fake mode should also fake the time it takes to fit and predict.
        Note: currently we are not compensating for the overhead of the simulation in anyway or form.
              TODO: this is future work

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
    fit_time_ : int, default=None
        Time in seconds needed to fit the original real classifier.
    predict_time_ : int, default=0
            Time the real model took to evaluate/infer the predictions
    confidences_ : ndarray, shape (n_samples, n_classes)
        The confidences on the test input samples. We expect the confidences to be in the order of the classes as
        they appear in the training ground truth. (E.g.: by np.unique(y) or by a label encoder)
    predict_proba_time_ : int, default=0
        Time the real model took to evaluate/infer the confidences
    label_encoder : bool, default=false
        Whether we need to apply encoding to the predictions.

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
                 predict_time_: int = 0, predict_proba_time_: int = 0, fit_time_: int = 0,
                 simulate_time: bool = False, label_encoder=False):
        self.simulate_time = simulate_time
        self.label_encoder = label_encoder

        self.fit_time_ = fit_time_

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
        sorter = np.argsort(self.oracle_index_)
        prediction_indices = sorter[np.searchsorted(self.oracle_index_,
                                                    test_data_hash_indices, sorter=sorter)]

        # Update labels similar to that in the meta model (FIXME, this wont work always...)
        if self.label_encoder:
            predictions = np.unique(self.predictions_, return_inverse=True)[1][prediction_indices]
        else:
            predictions = self.predictions_[prediction_indices]

        if self.simulate_time:
            time.sleep(self.predict_time_)

        return self.classes_[self.le_.transform(predictions)]

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
            confidences = check_array(confidences)

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
        # Currently "just" a hash and not a full checksum algorithm... might need to change this
        return hash(x.tobytes())  # Not consistent between runs (could not be stored), use hashlib for that


# -- Additional Functions for FakedClassifiers Usage
def initialize_fake_models(X_train, y_train, X_test, known_predictions, known_confidences, pre_fit_base_models,
                           base_models_with_names, label_encoder):
    # Expect the predictions/confidences on the whole meta-data as input. Whereby meta-data is the data passed to
    #   the ensemble method.

    faked_base_models = []

    for model_name in list(known_predictions):  # known predictions is a dataframe
        model_confidences = known_confidences[["confidence.{}.{}".format(class_name, model_name)
                                               for class_name in np.unique(y_train)]]
        model_predictions = known_predictions[model_name]
        fc = FakedClassifier(X_test, model_predictions, model_confidences, label_encoder=label_encoder)

        # -- Set fitted or not (sklearn vs. deslib)
        if pre_fit_base_models:
            if isinstance(X_train, pd.DataFrame) and isinstance(y_train, pd.Series):
                fc.fit(X_train.to_numpy(), y_train.to_numpy())
            elif isinstance(y_train, pd.Series):
                fc.fit(X_train, y_train.to_numpy())
            else:
                raise ValueError("Unsupported Types for X_train or y_train: " +
                                 "X_train type is {}; y_train type is {}".format(type(X_train), type(y_train)))

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
