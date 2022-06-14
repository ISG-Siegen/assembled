# Potentially useful metrics for evaluation wrapped in an easier to use object

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.validation import _check_y, check_array

from typing import Union, Callable
from abc import ABCMeta, abstractmethod


# -- Metric Utils
def make_metric(metric_func: Callable, metric_name: str, maximize: bool,
                classification: bool, always_transform_conf_to_pred: bool):
    """ Make a metric that has additional information

    Parameters
    ----------
    metric_func: Callable
        The metric function to call.
        We expect it to be metric_func(y_true, y_pred) with y_pred potentially being
        probabilities instead of classes.
    metric_name: str
        Name of the metric
    maximize: bool
        Whether to maximize the metric or not
    classification: bool
        If it is a classification metric or not
    always_transform_conf_to_pred: bool
        Set to Ture if the metric can not handle confidences and only accepts predictions (only for classification)
    """

    return AbstractMetric(metric_func, metric_name, maximize, classification,
                          always_transform_conf_to_pred)


class AbstractMetric:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, metric, name, maximize, classification, transform_conf_to_pred):
        self.metric = metric
        self.maximize = maximize
        self.name = name
        self.classification = classification
        self.transform_conf_to_pred = transform_conf_to_pred

    def __call__(self, y_true: Union[pd.DataFrame, np.ndarray], y_pred: Union[pd.DataFrame, np.ndarray],
                 to_loss: bool = False):
        """

        Parameters
        ----------
        y_true: array-like
            ground truth
        y_pred: array-like
            Either confidences/probabilities matrix (n_samples, n_classes) or prediction vector (n_samples, )
            If not classification, only prediction vector is allowed for now.
            If confidences, we expect the order of n_classes to be identical to the order of np.unique(y_true).
        to_loss: bool
            Whether to return the loss or not
        """

        # -- Input validation
        y_true = _check_y(y_true)

        if not self.classification:
            y_pred = _check_y(y_pred, y_numeric=True)
        else:
            if y_pred.ndim == 1:
                y_pred = _check_y(y_pred)
            elif y_pred.ndim == 2:
                y_pred = check_array(y_pred)

                # - Special case if metric can not handle confidences
                if self.transform_conf_to_pred:
                    y_pred = np.unique(y_true).take(np.argmax(y_pred, axis=1), axis=0)
            else:
                raise ValueError("y_pred has to many dimensions! Found ndim: {}".format(y_pred.ndim))

        # --- Call metric
        metric_value = self.metric(y_true, y_pred)

        # --- Return
        if to_loss:
            return self.to_loss(metric_value)

        return metric_value

    def to_loss(self, metric_value):
        """For now simply take the negative to get a loss."""
        if self.maximize:
            return -metric_value

        return metric_value


# -- Custom Metric Callers
def area_under_roc_curve(y_true, y_pred):
    """OpenML Variant of AUROC

    This was the only way we were able to re-create the scores found on OpenML.
    """

    classes_ = np.unique(y_true)
    y_true = label_binarize(y_true, classes=classes_)

    if y_pred.ndim == 1:
        # Case to handle non-confidence input
        # Transform predictions to fake confidences
        y_pred = label_binarize(y_pred, classes=classes_)
    elif len(classes_) <= 2:
        y_pred = y_pred[:, 1]

    return roc_auc_score(y_true, y_pred, multi_class="ovr", average="weighted")


# -- Metrics
BalancedAcc = make_metric(balanced_accuracy_score, "Balanced_Acc", True, True, True)
OpenMLAUROC = make_metric(area_under_roc_curve, "AUROC", True, True, False)
