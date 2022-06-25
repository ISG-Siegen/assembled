# Code Taken from here with adaptions to be usable:
# https://github.com/automl/auto-sklearn/blob/master/autosklearn/ensembles/abstract_ensemble.py

from abc import ABCMeta, abstractmethod
from typing import List, Optional
import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder


class AbstractEnsemble(object):
    """Abstract class to guarantee that we can use autosklearn's ensemble techniques and custom techniques.

    Parameters
    ----------
    base_models: List[Callable], List[sklearn estimators]
        The pool of fitted base models.
    predict_method: {"predict", "predict_proba"}, default="predict"
        Which predict method should be passed to the ensemble technique's fit.
    output_method: {"predict", "predict_proba"}, default="predict"
        Which output the ensemble will return.
    predict_method_ensemble_predict: {"predict", "predict_proba", None}, default=None
        Which predict method should be passed to the ensemble technique's predict. If None, the same method passed
        to ensemble fit is passed to ensemble predict.
    passthrough : bool, default=False
        When False, only the predictions or confidences of the base models will be passed to the ensemble fit and
        predict. When True, the original training data is passed additionally. The ensemble technique must support this!

    Attributes
    ----------
    le_ : LabelEncoder, object
        The label encoder created at :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, base_models, predict_method: str = "predict", output_method: str = "predict",
                 predict_method_ensemble_predict: Optional[str] = None, passthrough: bool = False):

        self.base_models = base_models
        self.predict_method = predict_method
        self.output_method = output_method
        self.passthrough = passthrough

        if predict_method_ensemble_predict is None:
            self.predict_method_ensemble_predict = predict_method
        else:
            self.predict_method_ensemble_predict = predict_method_ensemble_predict

    def fit(self, X, y):
        """Fitting the ensemble. To do so, we get the predictions of the base models and pass it to the ensemble's fit.

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
        if self.predict_method not in ["predict", "predict_proba"]:
            raise ValueError("Unknown predict method: {}".format(self.predict_method))
        if self.output_method not in ["predict", "predict_proba"]:
            raise ValueError("Unknown ensemble output method: {}".format(self.output_method))
            # Test if ensemble technique supports passthrough
        if self.passthrough and (not (hasattr(self, "supports_passthrough") and self.supports_passthrough is True)):
            raise ValueError("Passthrough is enabled but the ensemble technique does not support passthrough!")

        X, y = check_X_y(X, y)
        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_

        if self.passthrough:
            self.ensemble_passthrough_fit(X, self.base_models_predictions(X), y)
        else:
            self.ensemble_fit(self.base_models_predictions(X), y)

        return self

    def predict(self, X):
        """Predicting with the ensemble. To do so, we get the predictions of the base models and pass it to the ensemble

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Vector containing the class labels for each sample.
        """
        check_is_fitted(self, ["le_", "classes_"])
        X = check_array(X)

        if self.passthrough:
            ensemble_output = self.ensemble_passthrough_predict(X, self.base_models_predictions_for_ensemble_predict(X))
        else:
            ensemble_output = self.ensemble_predict(self.base_models_predictions_for_ensemble_predict(X))

        return self.transform_ensemble_prediction(ensemble_output)

    def base_models_predictions(self, X):
        if self.predict_method == "predict":
            return [bm.predict(X) for bm in self.base_models]
        else:
            return [bm.predict_proba(X) for bm in self.base_models]

    def base_models_predictions_for_ensemble_predict(self, X):
        # TODO, maybe merge with method above. IDK yet
        if self.predict_method_ensemble_predict == "predict":
            return [bm.predict(X) for bm in self.base_models]
        else:
            return [bm.predict_proba(X) for bm in self.base_models]

    def transform_ensemble_prediction(self, ensemble_output):
        if self.output_method == "predict":
            return ensemble_output
        else:
            return self._confidences_to_predictions(ensemble_output)

    def _confidences_to_predictions(self, confidences):
        return self.classes_[np.argmax(confidences, axis=1)]

    def oracle_predict(self, X, y):
        """Oracle Predict, predicting with knowing the ground truth. Only used by some methods for comparison

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Vector containing the class labels for each sample.
        """
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)

        ensemble_output = self.ensemble_oracle_predict(self.base_models_predictions_for_ensemble_predict(X), y)

        return self.transform_ensemble_prediction(ensemble_output)

    @abstractmethod
    def ensemble_fit(self, base_models_predictions: List[np.ndarray], labels: np.ndarray) -> 'AbstractEnsemble':
        """Fit an ensemble given predictions of base models and targets.

        base_models_predictions can either be the raw predictions or the confidences!

        Parameters
        ----------
        base_models_predictions : array of shape = [n_base_models, n_data_points, n_targets]
            n_targets is the number of classes in case of classification,
            n_targets is 0 or 1 in case of regression

        labels : array of shape [n_targets]

        Returns
        -------
        self

        """
        pass

    def ensemble_passthrough_fit(self, X, base_models_predictions: List[np.ndarray],
                                 labels: np.ndarray) -> 'AbstractEnsemble':
        """Fit an ensemble given predictions of base models and targets.

        base_models_predictions can either be the raw predictions or the confidences!

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        base_models_predictions : array of shape = [n_base_models, n_data_points, n_targets]
            n_targets is the number of classes in case of classification,
            n_targets is 0 or 1 in case of regression

        labels : array of shape [n_targets]

        Returns
        -------
        self

        """
        raise NotImplemented("ensemble_passthrough_fit is not implemented by every ensemble method!")

    @abstractmethod
    def ensemble_predict(self, base_models_predictions: List[np.ndarray]) -> np.ndarray:
        """Create ensemble predictions from the base model predictions.

        base_models_predictions can either be the raw predictions or the confidences!

        Parameters
        ----------
        base_models_predictions : array of shape = [n_base_models, n_data_points, n_targets]
            n_targets is the number of classes in case of classification,
            n_targets is 0 or 1 in case of regression

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Vector containing the class labels for each sample.
        """
        pass

    def ensemble_passthrough_predict(self, X, base_models_predictions: List[np.ndarray]) -> np.ndarray:
        """Create ensemble predictions from the base model predictions.

        base_models_predictions can either be the raw predictions or the confidences!

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        base_models_predictions : array of shape = [n_base_models, n_data_points, n_targets]
            n_targets is the number of classes in case of classification,
            n_targets is 0 or 1 in case of regression

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Vector containing the class labels for each sample.
        """
        raise NotImplemented("ensemble_passthrough_predict is not implemented by every ensemble method!")

    def ensemble_oracle_predict(self, base_models_predictions: List[np.ndarray], labels: np.ndarray) -> np.ndarray:
        """Predict with an oracle-like ensemble given predictions of base models and targets.

        base_models_predictions can either be the raw predictions or the confidences!

        Parameters
        ----------
        base_models_predictions : array of shape = [n_base_models, n_data_points, n_targets]
            n_targets is the number of classes in case of classification,
            n_targets is 0 or 1 in case of regression

        labels : array of shape [n_targets]

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Vector containing the class labels for each sample.
        """
        raise NotImplemented("ensemble_oracle_predict is not implemented by every ensemble method!")
