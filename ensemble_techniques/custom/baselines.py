import numpy as np
from typing import List
from ensemble_techniques.wrapper.abstract_ensemble import AbstractEnsemble
from assembledopenml.compatibility.openml_metrics import AbstractMetric
from sklearn.utils.validation import check_is_fitted


class VirtualBest(AbstractEnsemble):
    """Oracle-like Selector that can return predictions of the Virtual Best (VB) Model

    WARNING: This Ensemble Techniques assumes that the prediction of each base model is equal to the class with the
        highest confidence. If multiple classes have the same highest confidence, we treat it like all of these classes
        were predicted and are correct.

    Parameters
    ----------
    base_models: List[Callable], List[sklearn estimators]
        The pool of fitted base models.
    handle_no_correct: bool, default=True
        Whether to handle the case where no base model is able to correctly classify an instance.
        If false, we take the first base model in our base model list to predict for the instance.
            We could also take a random base model here but decided against it for the sake for making the VB static.
        If true, we take the base model with the highest confidence for the correct class.

    Attributes
    ----------
    selected_indices_: ndarray, shape (n_samples,)
        The selected indices per samples found during :meth:`ensemble_oracle_predict`.
        Used to determine selection performance.
    selected_indices_set_: ndarray, shape (n_samples, from 1 to n_base_models)
        The potential set of best models that could be selected per instance.
        Used to determine selection performance.
    """
    oracle_like = True

    def __init__(self, base_models, handle_no_correct=True):
        super().__init__(base_models, "predict_proba", "predict_proba")
        self.handle_no_correct = handle_no_correct

    def ensemble_fit(self, base_models_predictions: List[np.ndarray], labels: np.ndarray) -> 'AbstractEnsemble':
        return self

    def ensemble_predict(self, base_models_predictions: List[np.ndarray]) -> np.ndarray:
        raise NotImplemented("The VirtualBest Dynamic Selection Ensemble does not support a normal predict method.")

    def ensemble_oracle_predict(self, base_models_predictions: List[np.ndarray], labels: np.ndarray) -> np.ndarray:
        n_samples = base_models_predictions[0].shape[0]

        self.selected_indices_set_ = self._virtual_best_indices_sets(base_models_predictions, labels)
        self.selected_indices_ = [indices_set[0] for indices_set in self.selected_indices_set_]
        return np.array(base_models_predictions)[self.selected_indices_, np.arange(n_samples)]

    def _virtual_best_indices_sets(self, base_models_predictions, labels):
        """Create a list of sets with each set containing the best models' indices for an instance.

        For classification, we assume that an algorithm is a "best" algorithm for an instance if it can correctly
        classify the current instance. The probability of correct classification is not relevant for this.
        """
        label_indicator = self.le_.transform(labels)
        correct_predicted = np.array([[max(vals) == vals[i]
                                       for vals, i in zip(bm_confs, label_indicator)]
                                      for bm_confs in base_models_predictions])

        # This will, by default, return all base models' indices if no base model correctly classified an instance
        best_indices = [np.argwhere(bm_predictions == np.max(bm_predictions)).flatten().tolist()
                        for bm_predictions in correct_predicted.T]

        # -- Code for no VB Handling
        if self.handle_no_correct:
            # If for an instance in classification no example_algorithms was able to correctly classify the instance,
            #   we need to handle this.
            idx_no_best_model = np.argwhere(
                np.apply_along_axis(np.max, 0, correct_predicted) == False).flatten().tolist()  # == required!
            bm_confs = np.array(base_models_predictions)

            # Update values in list where needed
            for no_best_instance_idx in idx_no_best_model:
                correct_class_idx = np.where(self.classes_ == labels[no_best_instance_idx])[0][0]
                best_indices[no_best_instance_idx] = [np.argmax(bm_confs[:, no_best_instance_idx, correct_class_idx])]

        return best_indices


class SingleBest(AbstractEnsemble):
    """Single Best Selector

    Parameters
    ----------
    base_models: List[Callable], List[sklearn estimators]
        The pool of fitted base models.
    metric : AbstractMetric function, default=None
        The metric function that should be used to determine the single best algorithm
        Special format required due to OpenML's metrics and our usage.
    predict_method: {"predict", "predict_proba"}, default="predict"
        If "predict" is selected, we determine the SB by passing the raw predictions to the metric.
        If "predict_proba" is selected, we determine the SB by passing the confidences to the metric.
        If the metric can not handle confidences and "predict_proba" is selected, the behavior is identical to
            when "predict" would have been selected.

     Attributes
    ----------
    best_model_index_ : int
        The Index of Single Best Model found during :meth:`ensemble_fit`.
    selected_indices_: ndarray, shape (n_samples,)
        The selected indices per samples found during :meth:`ensemble_predict`. Used to determine selection performance.
    """

    def __init__(self, base_models, metric: AbstractMetric, predict_method="predict"):
        super().__init__(base_models, predict_method, "predict", "predict")
        self.predict_method = predict_method
        self.metric = metric

    def ensemble_fit(self, base_models_predictions: List[np.ndarray], labels: np.ndarray) -> AbstractEnsemble:
        """Find the single best algorithm and store it for later"""

        if not isinstance(self.metric, AbstractMetric):
            raise ValueError("The provided metric must be an instance of a AbstractMetric, "
                             "nevertheless it is {}({})".format(self.metric, type(self.metric)))

        if self.predict_method == "predict":
            performances = [self.metric(labels, y_pred=bm_prediction) for bm_prediction in base_models_predictions]
        else:
            performances = [self.metric(labels, y_conf=bm_prediction) for bm_prediction in base_models_predictions]

        if self.metric.maximize:
            self.best_model_index_ = np.argmax(performances)
        else:
            self.best_model_index_ = np.argmin(performances)

        return self

    def ensemble_predict(self, base_models_predictions: List[np.ndarray]) -> np.ndarray:
        """ Return the predictions of the Single Best"""

        check_is_fitted(self, ["best_model_index_"])

        n_samples = base_models_predictions[0].shape[0]
        self.selected_indices_ = np.full(n_samples, self.best_model_index_)

        return np.array(base_models_predictions)[self.selected_indices_, np.arange(n_samples)]
