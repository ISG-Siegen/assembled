import numpy as np
from typing import List
from ensemble_techniques.wrapper.abstract_ensemble import AbstractEnsemble
from ensemble_techniques.wrapper.abstract_dynamic_selector import AbstractDynamicSelector


class DSEmpiricalPerformanceModel(AbstractEnsemble, AbstractDynamicSelector):
    """A Simulated Dynamic Classifier Selector. Instead of aggregating prediction, this selects a prediction to use.
        Moreover, it always trains on the original X. Can speed up inference time and predictive quality.

    FIXME: Currently only support 1-EPM approach; only for classification; EPM must be a regressor
    WARNING: This Ensemble Techniques assumes that the prediction of each base model is equal to the class with the
        highest confidence. If multiple classes have the same highest confidence, we treat it like all of these classes
        were predicted and are correct.

    Parameters
    ----------
    base_models: List[Callable], List[sklearn estimators]
        The pool of fitted base models.
    epm : sklearn estimator/pipeline
        Empirical Performance Model (EPM) used to predict the performance of the base models. The predicted
        performance is used for dynamic selection. Here, the performance of a base model is equal to the error.
    epm_error: {"predict", "predict_proba"}, default="predict"
        What type of output from the base models is used to determine the error that is predicted by the EPM.
        If "predict" is selected, we determine the error for the EPM by using the raw predictions.
        If "predict_proba" is selected, we determine the error by using the confidences.
    wrong_prediction_penalty: int, default=0.2
        The penalty that is added to to the error of a predictor for an instance, if the predictor predicted
        the wrong class. This is only used if predict_method="predict_proba".
    ensemble_selection: bool, default=False
        If False, returns the prediction of the base model that is estimated to be the best base model for an instance.
        If True, returns the combined prediction of an ensemble of selected base models.
    ensemble_combination_method: {"voting", "soft_voting", "weighted_soft_voting"}, default="majority"
        The method used to combine the output of the selected ensemble.
            "voting": Majority vote with the raw predictions for classification.
            "soft_voting": Determine the predicted class by returning the class that has the highest confidences
                           after summing all base model's confidences (argmax of sums of predicted probabilities).
            "weighted_soft_voting": Same as 'soft_voting' but with a weighted sum based on the predicted error.

    Attributes
    ----------
    selected_indices_: ndarray, shape (n_samples,)
        The selected indices per samples found during :meth:`ensemble_predict`.
        Used to determine selection performance.
    selected_ensemble_indices_: ndarray, shape (n_samples,1-n_base_models)
        The indices of the selected ensemble per instance.
    """
    supports_passthrough = True

    def __init__(self, base_models, epm, epm_error: str = "predict_proba", wrong_prediction_penalty: int = 0.2,
                 ensemble_selection: bool = False, ensemble_combination_method="voting"):

        # -- Select output method (?maybe move this or do not have the else/raise case?)
        if (not ensemble_selection) or (ensemble_combination_method in ["voting"]):
            predict_method_ensemble_predict = "predict"
            output_method = "predict"
        elif ensemble_combination_method in ["soft_voting", "weighted_soft_voting"]:
            predict_method_ensemble_predict = "predict_proba"
            output_method = "predict_proba"
        else:
            raise ValueError("Unknown ensemble_combination_method: {}".format(ensemble_combination_method))

        # -- Init Parents
        AbstractEnsemble.__init__(self, base_models, predict_method=epm_error, output_method=output_method,
                                  passthrough=True, predict_method_ensemble_predict=predict_method_ensemble_predict)
        AbstractDynamicSelector.__init__(self, ensemble_selection=ensemble_selection,
                                         ensemble_combination_method=ensemble_combination_method)

        # -- Init Parameters
        self.epm = epm
        self.wrong_prediction_penalty = wrong_prediction_penalty

    def _build_epm_Y(self, base_models_predictions, labels) -> np.ndarray:
        if self.predict_method == "predict_proba":
            errors = [self._confidences_errors(bm_confs, labels) for bm_confs in base_models_predictions]
            return np.array(errors).T

        # Predict / Raw Predictions Case
        errors = [self._raw_predictions_errors(bm_predictions, labels) for bm_predictions in base_models_predictions]
        return np.array(errors).T

    def _confidences_errors(self, bm_confs, labels):
        """Error based on Confidences (a bit more complicated)

        To calculate the error based on the Confidences, we need to find the difference between  a base model's
        confidence for the correct class and 1.

        However, we can not forget to make the error about correct and wrong predictions, because a model can be
        confidently incorrect.

        Why? A counter example for just using the confidences for Algorithms A1,A2 and confidences for classes C1,C2,C3:
            A1: 0.3 (C1), 0.3 (C2), 0.4(C3) with C3 as GT, error 1-0.4 = 0.6
            A2: 0.5 (C1), 0.05 (C2), 0.45 (C3) with C3 as GT, error 1-0.45 = 0.55
                -> A2 has a lower error even tho wrong prediction

        To avoid this, I need to increase the error when the predictor predicted the wrong class.
        This increase must reflect "how far off" the predictor was.
        It must include a bias towards the correct prediction.

        We chose to add the difference and guarantee a minimal penalty.

        TODO: Can we learn the combination of errors?
        """

        # Collect values needed for the computation
        label_indicators = self.le_.transform(labels)
        wrong_classes_per_instance = np.array([[i for i in range(bm_confs.shape[1]) if i != x]
                                               for x in label_indicators])
        conf_wrong_classes = np.take_along_axis(bm_confs, wrong_classes_per_instance, axis=1)
        conf_correct_class = bm_confs[np.arange(len(bm_confs)), label_indicators]

        # Get the difference between the conf of the correct class and the conf of other classes
        additional_errors = np.subtract(conf_wrong_classes, conf_correct_class.reshape(-1, 1))
        # Ignore negative values, that is values where the correct class's conf is higher
        additional_errors[additional_errors < 0] = 0
        additional_errors = np.sum(additional_errors, axis=1)
        # Add a penalty to all wrong predicted instance based on confidence
        #   We know the predictor predicted wrong if the additional error is larger than 0.
        #       If the conf is equal for multiple classes, this is still zero and all good.
        #   We need this penalty to create (larger) gaps between the error values of different base models
        additional_errors[additional_errors > 0] += self.wrong_prediction_penalty

        # Combine collected information
        return 1 - conf_correct_class + additional_errors

    @staticmethod
    def _raw_predictions_errors(bm_predictions, labels):
        """Error for raw predictions is 0 if the base model correctly classifiy the instance and 1 otherwise."""
        return (bm_predictions != labels).astype(int)

    def ensemble_passthrough_fit(self, X, base_models_predictions: List[np.ndarray],
                                 labels: np.ndarray) -> 'AbstractEnsemble':
        """Train the EPM"""

        epm_Y = self._build_epm_Y(base_models_predictions, labels)
        self.epm.fit(X, epm_Y)

        return self

    def ensemble_passthrough_predict(self, X, base_models_predictions: List[np.ndarray]) -> np.ndarray:
        """Do dynamic selection using the EPM and return the selected base models' predictions"""

        self.epm_predictions_ = self.epm.predict(X)

        if not self.ensemble_selection:
            # We select the algorithm with the smallest error (we always want to w.l.o.g. minimize an error)
            self.selected_indices_ = np.argmin(self.epm_predictions_, axis=1)
        else:
            # We select the base models within in the 25% quantile of the error values
            self.selected_ensemble_indices_ = [np.flatnonzero(instance_error <= np.quantile(instance_error, 0.25))
                                               for instance_error in self.epm_predictions_]

        return self._ds_predict(base_models_predictions)

    def ensemble_fit(self, base_models_predictions: List[np.ndarray], labels: np.ndarray) -> 'AbstractEnsemble':
        raise NotImplemented("DSEmpiricalPerformanceModel requires passthrough!")

    def ensemble_predict(self, base_models_predictions: List[np.ndarray]) -> np.ndarray:
        raise NotImplemented("DSEmpiricalPerformanceModel requires passthrough!")
