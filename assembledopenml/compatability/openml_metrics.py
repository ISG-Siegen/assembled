import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from typing import Union


def no_conf_openml_area_under_roc_curve(y_ture: pd.Series, y_pred: pd.Series):
    class_labels = y_ture.unique().tolist()
    return openml_area_under_roc_curve(y_ture, y_pred, pd.DataFrame(), class_labels, use_binarized_y_pred=True)


def openml_area_under_roc_curve(y_ture: pd.Series, y_pred: pd.Series, y_conf: Union[pd.DataFrame, np.ndarray],
                                class_labels: list, use_binarized_y_pred=False):
    """
    Why so complicated?
    Well... the score on OpenML uses confidences and the setup as below. We basically simulate it here.
    """

    # Assumes y_conf columns are order like class labels

    n_classes = y_ture.nunique()
    y_true_bin = label_binarize(y_ture, classes=class_labels)
    y_conf_to_use = y_conf.copy()

    if use_binarized_y_pred:
        y_conf_to_use = label_binarize(y_pred, classes=class_labels)
    elif n_classes <= 2:
        # Binary case change
        if isinstance(y_conf_to_use, pd.DataFrame):
            y_conf_to_use = y_conf_to_use[class_labels[1]]
        else:
            y_conf_to_use = y_conf_to_use[:, 1]

    return roc_auc_score(y_true_bin, y_conf_to_use, multi_class="ovr", average="weighted",
                         labels=class_labels)
