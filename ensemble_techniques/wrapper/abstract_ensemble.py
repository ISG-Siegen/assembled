# Code Taken from here with (heavy) adaptions to be usable:
# https://github.com/automl/auto-sklearn/blob/master/autosklearn/ensembles/abstract_ensemble.py

from abc import ABCMeta, abstractmethod
from typing import List, Optional
import numpy as np
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder


class AbstractEnsemble(object):
    """Abstract class to guarantee that we can use autosklearn's ensemble techniques and custom techniques.

    During fit (for classification), we transform all labels into integers. This also happens for the predictions of
    base models.

    Parameters
    ----------
    base_models: List[Callable], List[sklearn estimators]
        The pool of fitted base models.
    predict_method: {"predict", "predict_proba"}, default="predict"
        Determine the predict method that is used to obtain the output of base models that is passed to an ensemble's
        fit method.
    output_method: {"predict", "predict_proba"}, default="predict"
        Which output the ensemble will return.
    predict_method_ensemble_predict: {"predict", "predict_proba", None}, default=None
        Determine the predict method that is used to obtain the output of base models that is passed to an ensemble's
        predict method.
        If None, the same method passed to ensemble fit is passed to ensemble predict.
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

        # Get the classes seen by the base model on the data they have been trained on.
        try:
            self.base_model_le_ = self.base_models[0].le_
        except AttributeError:
            # Most likely calibrated classifier or other wrapper
            self.base_model_le_ = self.base_models[0].base_estimator.le_

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

        # Check if self.classes_ differs
        if len(self.classes_) != len(self.base_model_le_.classes_):
            print("The number of seen classes differs for the base models and the ensemble.",
                  "We fix this by using the base model's label encoder.")
            # TO fix it, we use the label encoder of the base models
            self.le_ = self.base_model_le_
            self.classes_ = self.base_model_le_.classes_

        y_ = self.le_.transform(y)

        if self.passthrough:
            self.ensemble_passthrough_fit(X, self.base_models_predictions(X), y_)
        else:
            self.ensemble_fit(self.base_models_predictions(X), y_)

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
        check_is_fitted_for = []
        if hasattr(self, "fitted_attributes"):
            check_is_fitted_for = self.fitted_attributes

        check_is_fitted(self, ["le_", "classes_"] + check_is_fitted_for)
        X = check_array(X)

        if self.passthrough:
            ensemble_output = self.ensemble_passthrough_predict(X, self.base_models_predictions_for_ensemble_predict(X))
        else:
            ensemble_output = self.ensemble_predict(self.base_models_predictions_for_ensemble_predict(X))

        return self.transform_ensemble_prediction(ensemble_output)

    def predict_proba(self, X):
        """Predicting with the ensemble.
         To do so, we get the predictions of the base models and pass it to the ensemble

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples, n_classes)
            Vector containing the class probabilities for each class for each sample.
        """
        check_is_fitted(self, ["le_", "classes_"])
        X = check_array(X)

        if self.passthrough:
            raise NotImplemented("Predict Proba for Passthrough is not yet implemented.")
            ensemble_output = self.ensemble_passthrough_predict(X, self.base_models_predictions_for_ensemble_predict(X))
        else:

            if self.output_method == "predict_proba":
                ensemble_output = self.ensemble_predict(self.base_models_predictions_for_ensemble_predict(X))
            else:
                raise NotImplemented("Predict Proba for non-default proba predictors is not yet implemented.")
                ensemble_output = self.ensemble_predict_proba(self.base_models_predictions_for_ensemble_predict(X))

        return ensemble_output

    def base_models_predictions(self, X):
        if self.predict_method == "predict":
            return [self.le_.transform(bm.predict(X)) for bm in self.base_models]
        else:
            return [bm.predict_proba(X) for bm in self.base_models]

    def base_models_predictions_for_ensemble_predict(self, X):
        # TODO, maybe merge with method above. IDK yet
        if self.predict_method_ensemble_predict == "predict":
            return [self.le_.transform(bm.predict(X)) for bm in self.base_models]
        else:
            return [bm.predict_proba(X) for bm in self.base_models]

    def transform_ensemble_prediction(self, ensemble_output):
        if self.output_method == "predict":
            y_ = ensemble_output
        else:
            y_ = self._confidences_to_predictions(ensemble_output)

        return self.le_.inverse_transform(y_)

    def _confidences_to_predictions(self, confidences):
        return np.argmax(confidences, axis=1)

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
        y_ = self.le_.transform(y)

        ensemble_output = self.ensemble_oracle_predict(self.base_models_predictions_for_ensemble_predict(X), y_)

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
