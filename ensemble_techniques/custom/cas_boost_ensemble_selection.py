from typing import List, Optional, Union, Callable

import numpy as np

from ensemble_techniques.wrapper.abstract_ensemble import AbstractEnsemble
from assembledopenml.compatibility.openml_metrics import AbstractMetric
from ensemble_techniques.autosklearn.ensemble_selection import EnsembleSelection


# "TEST.TEST.TEST": {
#             "technique": CascadedBoostedEnsembleSelection,
#             "technique_args": {
#                                "metric": OpenMLAUROC(),
#                                "random_state": RandomState(rng_seed)},
#             "probability_calibration": "no"
#         }


class CascadedBoostedEnsembleSelection(AbstractEnsemble):
    """X

    X

    Parameters
    ----------
    base_models: List[Callable], List[sklearn estimators]
        The pool of fitted base models.
    metric: AbstractMetric
        The metric used to evaluate the models
    random_state: Optional[int | RandomState] = None
        The random_state used for ensemble selection.
        *   None - Uses numpy's default RandomState object
        *   int - Successive calls to fit will produce the same results
        *   RandomState - Truely random, each call to fit will produce
                          different results, even with the same object.
    """

    def __init__(self, base_models: List[Callable], metric: AbstractMetric,
                 random_state: Optional[Union[int, np.random.RandomState]] = None) -> None:
        super().__init__(base_models, "predict_proba", "predict_proba")
        self.metric = metric
        self.random_state = random_state

    def ensemble_fit(self, predictions: List[np.ndarray], labels: np.ndarray) -> AbstractEnsemble:

        n_base_models = len(predictions)

        # --- Lvl 0
        lvl_0_es = EnsembleSelection(None, 1, self.metric, bagging=False, mode="fast",
                                     random_state=self.random_state, use_best=False)
        lvl_0_es.ensemble_fit(predictions, labels)
        lvl_0_predictions = lvl_0_es.ensemble_predict(predictions)

        # --- Stuff for later
        weights_per_lvl = [lvl_0_es.weights_]
        loss_per_lvl = [lvl_0_es.train_loss_]
        error_per_lvl = [self._get_proba_error(labels, lvl_0_predictions)]

        # --- Iteration stuff
        iter_es_predictions = lvl_0_predictions
        iter_train_predictions = [p - lvl_0_predictions for p in predictions]

        drop_index = lvl_0_es.indices_[0]
        del iter_train_predictions[drop_index], lvl_0_es
        running_pred_sum = lvl_0_predictions

        for i in range(5):  # 5*10 +1 -> ensemble size 51

            iter_es = EnsembleSelection(None, 10, self.metric, bagging=False, mode="fast",
                                        random_state=self.random_state, use_best=False,
                                        predecessor_predictions=iter_es_predictions)
            # Fit and Predict
            iter_es.ensemble_fit(iter_train_predictions, labels)
            iter_es_predictions = iter_es.ensemble_predict(iter_train_predictions)
            running_pred_sum += iter_es_predictions

            # Get new rest preds
            if len(iter_train_predictions) < n_base_models:
                # Some values were dropped before
                iter_train_predictions.insert(drop_index, lvl_0_predictions)

                tmp_weights = list(iter_es.weights_)
                tmp_weights.insert(drop_index, 0)
                iter_es.weights_ = np.array(tmp_weights)

            iter_train_predictions = [last_pred - iter_es_predictions for last_pred in iter_train_predictions]

            # Store info for later
            weights_per_lvl.append(iter_es.weights_)
            loss_per_lvl.append(iter_es.train_loss_)
            error_per_lvl.append(self._get_proba_error(labels, running_pred_sum))

        self.weights_per_lvl_ = weights_per_lvl

        return self

    def _pseudo_residual(self, y_true, y_pred):

        lb = LabelBinarizer()
        lb.fit(y_true)
        transformed_labels = lb.transform(y_true)
        if transformed_labels.shape[1] == 1:
            transformed_labels = np.append(
                1 - transformed_labels, transformed_labels, axis=1
            )

        return np.sum(transformed_labels * (transformed_labels - y_pred), axis=1)

    def _get_proba_error(self, y_true, y_pred):
        # TODO decide on a loss metric to use here (do tests perhaps)
        # log_loss(labels, predictions)
        return proba_gap_loss(self.le_.transform(y_true), y_pred)

    def ensemble_predict(self, predictions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        # average = np.zeros_like(predictions[0], dtype=np.float64)
        tmp_predictions = np.zeros_like(predictions[0], dtype=np.float64)

        added_values = []
        pred_vector_through_time= []
        for weight_vector in self.weights_per_lvl_:
            mod_predictions = [p - tmp_predictions for p in predictions]
            np.add(tmp_predictions, self._es_predict(mod_predictions, weight_vector), out=tmp_predictions)


            added_values.append(self._es_predict(mod_predictions, weight_vector))
            pred_vector_through_time.append(np.copy(tmp_predictions))

        return tmp_predictions

    @staticmethod
    def _es_predict(predictions, weights):
        average = np.zeros_like(predictions[0], dtype=np.float64)
        tmp_predictions = np.empty_like(predictions[0], dtype=np.float64)

        # if predictions.shape[0] == len(self.weights_),
        # predictions include those of zero-weight models.
        if len(predictions) == len(weights):
            for pred, weight in zip(predictions, weights):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)

        # if prediction model.shape[0] == len(non_null_weights),
        # predictions do not include those of zero-weight models.
        elif len(predictions) == np.count_nonzero(weights):
            non_null_weights = [w for w in weights if w > 0]
            for pred, weight in zip(predictions, non_null_weights):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)

        # If none of the above applies, then something must have gone wrong.
        else:
            raise ValueError("The dimensions of ensemble predictions"
                             " and ensemble weights do not match!")
        del tmp_predictions
        return average


# -- Proba Metrics
from sklearn.utils.validation import check_array, check_consistent_length
from sklearn.preprocessing import LabelBinarizer


def _weighted_sum(sample_score, sample_weight, normalize=False):
    if normalize:
        return np.average(sample_score, weights=sample_weight)
    elif sample_weight is not None:
        return np.dot(sample_score, sample_weight)
    else:
        return sample_score.sum()


def log_loss(y_true, y_pred, *, eps=1e-15, normalize=True, sample_weight=None):
    r"""Log loss, aka logistic loss or cross-entropy loss.

    .. math::
        L_{\log}(y, p) = -(y \log (p) + (1 - y) \log (1 - p))


    """
    y_pred = check_array(y_pred, ensure_2d=False)
    check_consistent_length(y_pred, y_true, sample_weight)

    lb = LabelBinarizer()
    lb.fit(y_true)

    if len(lb.classes_) == 1:
        raise ValueError(
            "The labels array needs to contain at least two "
            "labels for log_loss, "
            "got {0}.".format(lb.classes_)
        )

    transformed_labels = lb.transform(y_true)

    if transformed_labels.shape[1] == 1:
        transformed_labels = np.append(
            1 - transformed_labels, transformed_labels, axis=1
        )

    # Clipping
    y_pred = np.clip(y_pred, eps, 1 - eps)

    # If y_pred is of single dimension, assume y_true to be binary
    # and then check.
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)

    # Check if dimensions are consistent.
    transformed_labels = check_array(transformed_labels)

    if len(lb.classes_) != y_pred.shape[1]:
        raise ValueError(
            "The number of classes in labels is different "
            "from that in y_pred. Classes found in "
            "labels: {0}".format(lb.classes_)
        )

    # Renormalize
    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
    loss = -(transformed_labels * np.log(y_pred)).sum(axis=1)

    return _weighted_sum(loss, sample_weight, normalize)


def proba_gap_loss(label_indicators, bm_confs, wrong_prediction_penalty=0.4):
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
    additional_errors[additional_errors > 0] += wrong_prediction_penalty

    # Combine collected information
    loss = 1 - conf_correct_class + additional_errors
    return _weighted_sum(loss, None, True)



# TODO ideas
#   fix oscillating bug from predict and figure out what hte fian lweights will be with the minus  
#   smaller steps =5? what happens if steps=1?
#   what is the acutal resulting weights vector in the end?