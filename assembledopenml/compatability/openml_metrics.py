import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize
from typing import Union, Optional
from abc import ABCMeta, abstractmethod
from sklearn.utils.validation import _check_y, check_array


class AbstractMetric:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, maximize, name):
        self.maximize = maximize
        self.name = name

    @abstractmethod
    def __call__(self, y_ture: Union[pd.DataFrame, np.ndarray], y_pred: Union[pd.DataFrame, np.ndarray] = None,
                 y_conf: Optional[Union[pd.DataFrame, np.ndarray]] = None):
        pass


class Accuracy(AbstractMetric):
    """Wrapper for a Metric to handle confidences/predict_proba input."""

    def __init__(self):
        super().__init__(True, "Accuracy")

    def __call__(self, y_ture: Union[pd.DataFrame, np.ndarray], y_pred: Union[pd.DataFrame, np.ndarray] = None,
                 y_conf: Optional[Union[pd.DataFrame, np.ndarray]] = None):

        # -- Input validation
        y_ture = _check_y(y_ture)
        self.classes_ = np.unique(y_ture)

        # -- Call functions
        if y_conf is None:
            y_pred = _check_y(y_pred)
            return accuracy_score(y_ture, y_pred)
        elif y_pred is None:
            y_conf = check_array(y_conf)
            y_pred = np.array(self.classes_)[np.argmax(y_conf, axis=1)]
            return accuracy_score(y_ture, y_pred)
        else:
            raise ValueError("Either y_pred or y_conf must be not None!")


class OpenMLAUROC(AbstractMetric):

    def __init__(self):
        super().__init__(True, "AUROC")

    def __call__(self, y_ture: Union[pd.DataFrame, np.ndarray], y_pred: Union[pd.DataFrame, np.ndarray] = None,
                 y_conf: Optional[Union[pd.DataFrame, np.ndarray]] = None):

        # -- Input validation
        y_ture = _check_y(y_ture)
        self.classes_ = np.unique(y_ture)

        # -- Call functions
        if y_conf is None:
            y_pred = _check_y(y_pred)
            return self.area_under_roc_curve(y_ture, y_pred, None, True)
        elif y_pred is None:
            y_conf = check_array(y_conf)
            return self.area_under_roc_curve(y_ture, None, y_conf, False)
        else:
            raise ValueError("Either y_pred or y_conf must be not None!")

    def area_under_roc_curve(self, y_ture, y_pred, y_conf, use_binarized_y_pred):
        """
        Why so complicated?
        Well... the score on OpenML uses confidences and the setup as below. We basically simulate it here.
        """

        y_true_bin = label_binarize(y_ture, classes=self.classes_)

        if use_binarized_y_pred:
            # Handle no confidences given
            y_conf_to_use = label_binarize(y_pred, classes=self.classes_)
        elif len(self.classes_) <= 2:
            y_conf_to_use = y_conf[:, 1]
        else:
            y_conf_to_use = y_conf

        return roc_auc_score(y_true_bin, y_conf_to_use, multi_class="ovr", average="weighted",
                             labels=self.classes_)
