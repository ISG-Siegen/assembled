import pandas as pd
import numpy as np
import os
import json

from assembled.compatibility.faked_classifier import probability_calibration_for_faked_models, _initialize_fake_models
from typing import List, Tuple, Optional, Callable, Union
from sklearn.model_selection import train_test_split


class MetaTask:
    """Metatask, a meta version of a normal machine learning task

    The Metatask contains the predictions and confidences (e.g. sklearn's predict_proba) of specific base models
    and the data of the original (OpenML) task. Moreover, additional side information are captured.
    This object is filled via functions and thus it is empty initially.

    In the current version, we manage Metatasks as a meta_dataset (represented by a DataFrame) that contains all
    instance related data and meta_data (represented by a dict/json) that contains side information.
    """

    def __init__(self):
        # -- for dataset init
        self.dataset = None
        self.dataset_name = None
        self.target_name = None
        self.class_labels = None
        self.feature_names = None
        self.cat_feature_names = None
        self.task_type = None
        self.is_classification = None
        self.is_regression = None
        self.openml_task_id = None
        self.folds = None  # An array where each value represents the fold of an instance (starts from 0)

        # -- For Base Models
        self.predictions_and_confidences = pd.DataFrame()
        self.predictors = []  # List of predictor names (aka base models)
        self.predictor_descriptions = {}  # Predictor names to descriptions (description could be the configuration)
        self.bad_predictors = []  # List of predictor names that have ill-formatted predictions or wrong data
        self.fold_predictors = []
        self.predictor_corruptions_details = {}
        self.confidences = []  # List of column names of confidences
        self.validation_predictions_and_confidences = pd.DataFrame()
        self.use_validation_data = False
        self.validation_indices = {}

        # -- Selection constrains (used to find the data of this metatask)
        self.selection_constraints = {}

        # -- Other
        self.supported_task_types = {"classification", "regression"}
        self.confidence_prefix = "confidence"
        self.fold_postfix = "fold"
        self.fold_predictor_prefix = "FP"

        # -- Randomness (Experimental)
        self.random_int_seed_outer_folds = None
        self.random_int_seed_inner_folds = None

        # -- Backwards Compatibility
        self.missing_metadata_in_file = []

    # -- Other Object Related Functions
    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False

        self_vals = self.__dict__
        other_vals = other.__dict__

        for attr in self_vals:

            if attr not in other_vals:
                return False

            if isinstance(self_vals[attr], pd.DataFrame):
                try:
                    # Not using equals because of floating point errors
                    pd.testing.assert_frame_equal(self_vals[attr], other_vals[attr],
                                                  check_categorical=False)
                    # Check_categorical is false because task build from raw openml data can have
                    #   strange categorical values that do not exist in the dataset
                    #       (like a category that never appears in the column or an order)

                except AssertionError:
                    return False
            elif isinstance(self_vals[attr], np.ndarray):
                if not np.array_equal(self_vals[attr], other_vals[attr]):
                    return False
            elif isinstance(self_vals[attr], dict):
                # Check for dicts (required due to ndarray)
                self_dict = self_vals[attr]
                other_dict = other_vals[attr]
                if self_dict.keys() != other_dict.keys():
                    return False

                for k in self_dict.keys():
                    self_dict_val = self_dict[k]
                    if isinstance(self_dict_val, np.ndarray):
                        if not np.array_equal(self_dict_val, other_dict[k]):
                            return False
                    else:
                        if self_dict_val != other_dict[k]:
                            return False
            else:
                if other_vals[attr] != self_vals[attr]:
                    return False

        return True

    # -- Extended Side Information
    @property
    def n_classes(self):
        return len(self.class_labels)

    @property
    def n_instances(self):
        return len(self.dataset)

    @property
    def n_predictors(self):
        """Important: this contains all predictors. Hence, this includes predictors from each fold.

        Use n_predictors_per_fold to get a more accurate picture of predictors per fold.
        """
        return len(self.predictors)

    @property
    def n_predictors_per_fold(self):
        return [len(self.get_predictors_for_fold(i)) for i in range(self.max_fold + 1)]

    def get_predictors_for_fold(self, fold_idx):
        predictions_columns = []
        for pred in self.predictors:
            if pred not in self.fold_predictors:
                # Not a fold predictor
                predictions_columns.append(pred)
            else:
                if pred.startswith(self.to_fold_predictor_name("", fold_idx)):
                    # Fold predictor for the current fold
                    predictions_columns.append(pred)
        return predictions_columns

    @property
    def max_fold(self):
        return int(self.folds.max())

    # -- Properties related to meta-data
    @property
    def meta_data(self):
        meta_data = {md_k: getattr(self, md_k) for md_k in self.meta_data_keys}

        return meta_data

    @property
    def meta_data_keys(self):
        return ["openml_task_id", "dataset_name", "target_name", "class_labels", "predictors", "confidences",
                "predictor_descriptions", "bad_predictors", "fold_predictors", "confidence_prefix", "feature_names",
                "cat_feature_names", "selection_constraints", "task_type", "predictor_corruptions_details",
                "use_validation_data", "random_int_seed_outer_folds", "random_int_seed_inner_folds",
                "folds", "fold_postfix", "fold_predictor_prefix", "validation_indices"]

    # -- Properties related to meta_dataset
    @property
    def meta_dataset(self):
        return pd.concat([self.dataset, self.predictions_and_confidences, self.validation_predictions_and_confidences],
                         axis=1)

    @property
    def ground_truth(self) -> pd.Series:
        return self.dataset[self.target_name]

    @ground_truth.setter
    def ground_truth(self, value):
        self.dataset[self.target_name] = value

    @property
    def non_cat_feature_names(self):
        return [f for f in self.feature_names if f not in self.cat_feature_names]

    # -- Column Names Management for the different parts of the meta-dataset
    @property
    def pred_and_conf_cols(self):
        return self.get_pred_and_conf_cols(self.predictors)

    def get_pred_and_conf_cols(self, predictor_names):
        # return relevant predictor and confidence columns in a specific order
        return [ele for slist in [[col_name, *[self.to_confidence_name(col_name, n) for n in self.class_labels]]
                                  for col_name in predictor_names] for ele in slist]

    def get_conf_cols(self, predictor_names):
        return [self.to_confidence_name(col_name, n) for n in self.class_labels for col_name in predictor_names]

    @property
    def validation_predictions_columns(self):
        return self.get_validation_predictions_columns(self.predictors)

    def get_validation_predictions_columns(self, predictors):
        if not self.use_validation_data:
            return []

        return_val = []
        for pred_name in predictors:
            if pred_name in self.fold_predictors:
                # Special case for fold predictors which only have validation data for the respective fold
                return_val.append(self.to_validation_predictor_name(pred_name,
                                                                    self.fold_predictor_name_to_fold_idx(pred_name)))
            else:
                return_val.extend([self.to_validation_predictor_name(pred_name, fold_idx)
                                   for fold_idx in range(self.max_fold + 1)])

        return return_val

    @property
    def validation_confidences_columns(self):
        return self.get_validation_confidences_columns(self.predictors)

    def get_validation_confidences_columns(self, predictors):
        if not self.use_validation_data:
            return []

        return [self.to_confidence_name(p_name, n) for n in self.class_labels
                for p_name in self.get_validation_predictions_columns(predictors)]

    def to_fold_predictor_name(self, predictor_name, fold_idx):
        return "{}{}.{}".format(self.fold_predictor_prefix, fold_idx, predictor_name)

    def to_validation_predictor_name(self, predictor_name, fold_idx):
        return "{}.{}{}".format(predictor_name, self.fold_postfix, fold_idx)

    def from_validation_predictor_name_to_predictor_name(self, val_predictor_name):
        return val_predictor_name.rsplit(self.fold_postfix, 1)[0][:-1]

    def to_confidence_name(self, predictor_name, class_name):
        return "{}.{}.{}".format(self.confidence_prefix, class_name, predictor_name)

    def fold_predictor_name_to_fold_idx(self, pred_name) -> str:
        if pred_name not in self.fold_predictors:
            raise ValueError("Unknown Fold Predictor Name. Got: {}".format(pred_name))

        return pred_name[len(self.fold_predictor_prefix):].split(".", 1)[0]

    # --- Function to build a Metatask given data
    def init_dataset_information(self, dataset: pd.DataFrame, target_name: str, class_labels: List[str],
                                 feature_names: List[str], cat_feature_names: List[str], task_type: str,
                                 openml_task_id: int, folds_indicator: np.ndarray, dataset_name: str):
        """Fill dataset information and basic task information

        Parameters
        ----------
        dataset : pd.DataFrame
            The original dataset from OpenML.
        target_name: str
            The name of the target column of the dataset.
        class_labels: List[str]
            The class labels of the dataset.
        feature_names: List[str]
            The names of the feature columns of the dataset. (Including categorical features).
        cat_feature_names: List[str]
            The name of the categorical feature columns of the dataset.
        task_type: {"classification", "regression"}
            String determining the task type.
        openml_task_id: int
            OpenML Task ID, Use a negative number like -1 if no OpenML Task. This will be the tasks ID / name.
        folds_indicator: np.ndarray
            Array of length (n_samples,) indicating the folds for each instances (starting from 0)
            We do not support hold-out validation currently.
            Please be aware, the order of instances in the dataset should be equal to the order of the instances in
            the fold_indicator. We do not check this / can not check this.
        dataset_name: str
            Name of the dataset
        """

        # -- Input check for arrays (which would not be serializable)
        if isinstance(class_labels, (np.ndarray, pd.Series)):
            class_labels = class_labels.tolist()
        if isinstance(feature_names, (np.ndarray, pd.Series)):
            feature_names = feature_names.tolist()
        if isinstance(cat_feature_names, (np.ndarray, pd.Series)):
            cat_feature_names = cat_feature_names.tolist()

        if any(not isinstance(x, str) for x in class_labels):
            raise ValueError("Class labels must be strings!")

        # -- Save Data
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.target_name = target_name
        self.class_labels = sorted(class_labels)
        self.feature_names = feature_names
        self.cat_feature_names = cat_feature_names
        self.openml_task_id = openml_task_id
        self.read_folds(folds_indicator)

        # -- Re-order dataset for common standard
        self.dataset = self.dataset[feature_names + [self.target_name]]

        # Handle task type
        self.task_type = task_type
        if task_type not in self.supported_task_types:
            raise ValueError("Unsupported Task Type Str '{}' was found.".format(task_type))
        self.is_classification = task_type == "classification"
        self.is_regression = task_type == "regression"

        self._check_and_init_ground_truth()

        self._dataset_sanity_checks()

    def _check_and_init_ground_truth(self):
        # -- Process and Check ground truth depending on task type
        ground_truth = self.ground_truth

        if self.is_classification:

            # Handle String / Bool Reading Problem
            if set(ground_truth.unique().tolist()) - set(self.class_labels):
                # This means the labels of the ground truth col and provided class label names are not identical
                # One Solution: The Dataset's ground truth columns format is wrong.

                # - Check Special Cases for Bad String formatting
                if pd.api.types.is_bool_dtype(ground_truth.dtype):
                    # The reason is boolean formatting. Trying to fix this.
                    ground_truth = ground_truth.astype("str")

                    if all(l_str.isupper() for l_str in self.class_labels):
                        ground_truth = ground_truth.str.upper()
                    elif all(l_str.islower() for l_str in self.class_labels):
                        ground_truth = ground_truth.str.lower()
                    elif not set(ground_truth.unique().tolist()) - set(self.class_labels):
                        # Setting it to str type fixed the problem already...
                        pass
                    else:
                        raise ValueError("Unknown Str format of original labels in OpenML. ",
                                         "Unable to fix boolean formatting bug automatically. ",
                                         "For Task {} with labels {}.".format(self.openml_task_id, self.class_labels))
                else:
                    raise ValueError("Unknown Bad Format of Ground Truth Column for this Dataset. ",
                                     "For Task {} with labels {}.".format(self.openml_task_id, self.class_labels))

                # Final Re-Check if fix worked
                if set(ground_truth.unique().tolist()) - set(self.class_labels):
                    raise ValueError("Formatting Fix did not work. ",
                                     "The Dataset's ground truth columns are still ill-formatted.")

            # Set target to cat
            ground_truth = ground_truth.astype("category")
            ground_truth = ground_truth.cat.set_categories(self.class_labels)
        else:
            # We have nothing to check for Regression // No checks implemented
            pass

        self.ground_truth = ground_truth

    # - Loading Predictors
    def add_predictor(self, predictor_name: str, predictions: np.ndarray, confidences: Optional[np.ndarray] = None,
                      conf_class_labels: Optional[List[str]] = None, predictor_description: Optional[str] = None,
                      bad_predictor: bool = False, corruptions_details: Optional[dict] = None,
                      validation_data: Optional[List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = None,
                      fold_predictor: bool = False, fold_predictor_idx: Optional[int] = None):
        """Add a new predictor (base model) to the metatask

        Parameters
        ----------
        predictor_name: str
            name of the predictor; must be unique within a given metatask! (Overwriting is not supported yet)
        predictions: array-like, (n_samples,) or (n_fold_samples,)
            Cross-val-predictions for the predictor that correspond to the fold_indicators of the metatask.
            If fold_predictor, it only needs to contain the predictors for the specific fold.
        confidences: array-like, Optional, (n_samples, n_classes) or (n_fold_samples, n_classes), default=None
            Confidences of the prediction. If None, due to Regression tasks or no confidences, use default confidences.
            If fold_predictor, it only needs to contain the confidences for the specific fold.
        conf_class_labels: List[str], Optional, default = None
            The order of the class labels in that the confidences are passed to this function.
            It must contain the same labels as self.class_labels but can be in a different order!
            This is very important to get right, else all confidence values will be wrongly used.
        predictor_description: str, Optional, default=None
            A short description of the predictor (e.g., the configuration as a string like in OpenML).
            If None, an automatic description is created.
        bad_predictor: bool, default=False
            Set whether this predictor has some issue in its data or if any other reasons makes it bad (e.g. bad
            performance).
            Here, bad means that the metatasks should not use such predictors or allows them to be filtered later on.
        corruptions_details: dict, default=None
            A dict containing more details on why the predictor is bad or any other information you would want to keep
            for later on the predictor. E.g., Assembled-OpenML uses this to store whether the confidences values of the
            predictor had to be fixed.
        validation_data: List[Tuple[int, array-like, array-like, array-like]], default=None
            The validation data of the predictor for all relevant folds. If fold_predictor, the list only needs to
            contain the entry for the fold_predictor's fold.
            We assume an input of a list of lists which contain: Tuple[the fold index, the predictions on the validation
            data, confidences on the validation data, indices of the validation data].
            Validation data can be all training data instances or a subset of all training data instances.
            We expect that the instances used for validation are identical for all predictors of a metatask.
        fold_predictor: bool, default=False
            Whether the predictor that is to be added, is only for a specific fold. If False, we assume the prediction
            data of the predictor was computed for all folds while the predictor's configuration was consistent across
            folds. If True, we add the predictor and its data only for one specific fold. That fold has to be specified
            by fold_predictor_idx.
        fold_predictor_idx: int, default=None
            Required if fold_predictor is True. Specifies the fold index for which this predictors has been
            computed.
        """

        if confidences is None:
            # TODO add code for regression and non-confidences data (fake confidences)
            #   Goal would be to also have confidences for regression (CI interval or std)
            raise NotImplementedError("We currently require confidences for a predictor! Sorry...")

        # -- Input Checks
        if conf_class_labels is not None:
            if not set(conf_class_labels) == set(self.class_labels):
                raise ValueError("Unknown class labels as input. Expected: {}; Got: {}".format(self.class_labels,
                                                                                               conf_class_labels))
            class_labels_to_use = conf_class_labels
        else:
            class_labels_to_use = self.class_labels

        if predictor_description is None:
            raise NotImplementedError("Automatic Description building is not yet supported. "
                                      "Please set the description by hand.")

        if validation_data is None and self.use_validation_data:
            raise ValueError("Validation data is required if at least one other base model has validation!")

        if (validation_data is not None) and (self.n_predictors != 0) and (not self.use_validation_data):
            raise ValueError("You are trying to add a predictor with validation data, but previous predictors "
                             "do not have validation data. We require this to be consistent - all predictors must have "
                             "validation data or no predictor must have validation data.")

        if (not fold_predictor) and (predictions.shape[0] != self.n_instances):
            raise ValueError("Predictions are not for all instances of the dataset. "
                             "Have {} instances, got {} predictions".format(self.n_instances, len(predictions)))
        if confidences is not None:
            if (not fold_predictor) and (confidences.shape[0] != self.n_instances):
                raise ValueError("Confidences are not for all instances of the dataset. "
                                 "Have {} instances, got {} confidences".format(self.n_instances, len(confidences)))
            if confidences.shape[1] != self.n_classes:
                raise ValueError("Confidences are not for all classes of the dataset. "
                                 "Have {} classes, got {} confidences columns".format(self.n_classes,
                                                                                      confidences.shape[1]))

        if fold_predictor and (fold_predictor_idx is None):
            raise ValueError("We require the fold index via the parameter fold_predictor_idx if the predictors and "
                             "its data is only for a specific fold.")

        # -- Preliminary Work
        if fold_predictor:
            # update predictor name
            predictor_name = self.to_fold_predictor_name(predictor_name, fold_predictor_idx)
            self.fold_predictors.append(predictor_name)

        if predictor_name in self.predictors:
            raise ValueError("The name of the predictor already exist. Can not overwrite."
                             " The name must be unique. Got: {}".format(predictor_name))

        conf_col_names = [self.to_confidence_name(predictor_name, n) for n in class_labels_to_use]

        # Add predictor data to metadata
        self.predictors.append(predictor_name)
        self.confidences.extend(conf_col_names)
        self.predictor_descriptions[predictor_name] = predictor_description

        if bad_predictor:
            self.bad_predictors.append(predictor_name)
        if corruptions_details is not None:
            self.predictor_corruptions_details[predictor_name] = corruptions_details

        # -- Add normal prediction data
        if not fold_predictor:
            self.predictions_and_confidences = self._get_prediction_data(predictions, confidences, predictor_name,
                                                                         class_labels_to_use, conf_col_names,
                                                                         self.predictions_and_confidences.copy())
        else:
            other_indices, predicted_on_indices = self.get_indices_for_fold(fold_predictor_idx, return_indices=True)

            self.predictions_and_confidences = self._get_prediction_data(predictions, confidences, predictor_name,
                                                                         class_labels_to_use, conf_col_names,
                                                                         self.predictions_and_confidences.copy(),
                                                                         fold_data=True,
                                                                         fold_data_arguments={
                                                                             "non_used_indices": other_indices,
                                                                             "used_indices": predicted_on_indices
                                                                         })

        # -- Add validation data
        if validation_data is not None:
            # (Re-)Mark validation data usage
            self.use_validation_data = True
            tmp_val_data = self.validation_predictions_and_confidences.copy()

            for fold_idx, val_preds, val_confs, val_indices in validation_data:
                # -- Fold preliminaries
                save_name = self.to_validation_predictor_name(predictor_name, fold_idx)
                val_conf_col_names = [self.to_confidence_name(save_name, n) for n in class_labels_to_use]
                potential_val_indices, non_val_indices = self.get_indices_for_fold(fold_idx, return_indices=True)

                # -- Verify that validation indices are identical to previous seen indices of this fold
                known_val_indices_for_fold = self.validation_indices.get(fold_idx, None)
                if known_val_indices_for_fold is None:
                    self.validation_indices[fold_idx] = val_indices
                else:
                    if (np.sort(val_indices) != np.sort(known_val_indices_for_fold)).any():
                        raise ValueError("Input Validation indices for fold {} ".format(fold_idx)
                                         + "are not identical to previous seen validation indices for this fold.")

                # -- Make sure val_indices is an array
                if (not isinstance(val_indices, np.ndarray)) and isinstance(val_indices, list):
                    val_indices = np.array(val_indices)

                # -- Sanity Check: validation indices can not be indices from the test set!
                if np.intersect1d(val_indices, non_val_indices).size != 0:
                    raise ValueError("Validation indices contain indices from the test set of the current fold!")

                # -- Find elements that are not in the validation data (e.g. in the case of holdout)
                additional_non_val_indices = np.setdiff1d(potential_val_indices, val_indices)
                non_val_indices = np.append(non_val_indices, additional_non_val_indices)

                # Get fold prediction data
                tmp_val_data = self._get_prediction_data(val_preds, val_confs, save_name,
                                                         class_labels_to_use, val_conf_col_names,
                                                         tmp_val_data,
                                                         fold_data=True,
                                                         fold_data_arguments={
                                                             "non_used_indices": non_val_indices,
                                                             "used_indices": val_indices
                                                         })

            # -- Post-processing of fold data
            # Re-order validation data to have identical order at all times
            self.validation_predictions_and_confidences = tmp_val_data[self.validation_predictions_columns +
                                                                       self.validation_confidences_columns]

    def _get_prediction_data(self, predictions, confidences, predictor_name, class_labels_to_use, conf_col_names,
                             current_prediction_data, fold_data: bool = False,
                             fold_data_arguments: dict = None):
        """Checks are formats prediction data where needed. Return contacted prediction data.

        Relevant Parameters
        ----------
        fold_data: bool, default=False
            If True, we treat the prediction data like it has been computed only on a subset of all instances.
            To do so, we fill the prediction data with nan values for every not-predicted-for instance.
            This is later filtered by the evaluation code such that the nan values are ignored.
        fold_data_arguments: dict, default=None
            If fold_data is True, we require a dict of the form
            {"non_used_indices": indices_1, "used_indices": indices_2}. Where indices_1 contains all the indices for
            which the prediction data has not predictions and indices_2 all the indices for which the prediction data
            has indices.
        """

        # -- Handle Predictions
        # Predictions to Series with correct name
        if isinstance(predictions, pd.Series):
            predictions = predictions.rename(predictor_name)
        else:
            # Assume it is an array-like, if not duck-typing will tell us
            predictions = pd.Series(predictions, name=predictor_name)

        # -- Handle Confidences
        # Confidences to DataFrame
        if isinstance(confidences, pd.DataFrame):
            re_name = {confidences.columns[i]: n for i, n in enumerate(conf_col_names)}
            confidences = confidences.rename(re_name, axis=1)
        else:
            # Assume it is an array-like, if not duck-typing will tell us
            confidences = pd.DataFrame(confidences, columns=conf_col_names)

        # -- Unify Predictions data
        pred_data = pd.concat([predictions, confidences], axis=1)

        # -- Special Handling for validation data
        if fold_data:
            # Fill indices for the part of the validation data does not cover
            non_used_indices = fold_data_arguments["non_used_indices"]
            used_indices = fold_data_arguments["used_indices"]

            # Build filler data
            val_pred_data_filler = pd.DataFrame(np.full((len(non_used_indices), len(pred_data.columns)), np.nan),
                                                columns=pred_data.columns)

            # Add Tmp idx for later
            pred_data["tmp_idx"] = used_indices
            val_pred_data_filler["tmp_idx"] = non_used_indices

            # Fill pred data with missing instances
            pred_data = pd.concat([pred_data, val_pred_data_filler], axis=0).sort_values(
                by="tmp_idx").drop(columns=["tmp_idx"]).reset_index(drop=True)

        # -- Handle Dtype for all predictions if classification
        if self.is_classification:
            pred_data[predictor_name] = pred_data[predictor_name].astype('category').cat.set_categories(
                class_labels_to_use)

        # -- Concat and Verify integrity of index
        tmp_data = pd.concat([current_prediction_data, pred_data], axis=1).reset_index()
        if sum(tmp_data.index != tmp_data["index"]) != 0:
            raise ValueError("Something went wrong with the index of the predictions data!")
        tmp_predictions_and_confidences = tmp_data.drop(columns=["index"])

        return tmp_predictions_and_confidences

    # - Loading Side Information
    def read_selection_constraints(self, selection_constraints):
        """Fill the constrains used to build the metatask.

        This only updates but does not overwrite existing keys.

        Parameters
        ----------
        selection_constraints : dict
            A dict containing the names and values for selection constraints
        """
        self.selection_constraints.update(selection_constraints)

    def read_randomness(self, random_int_seed_outer_folds: Union[int, str],
                        random_int_seed_inner_folds: Optional[int] = None):
        """

        Parameters
        ----------
        random_int_seed_outer_folds: int or str
            The random seed (integer) used to create the folds. If not available, pass a short description why.
        random_int_seed_inner_folds: int, default=None
            We assume that the splits used to get the validation data were generated by some controlled randomness.
            That is, a RandomState object initialized with a base seed was (re-)used each fold to generate the splits.
            Here, we want to have the base seed to store

        """
        self.random_int_seed_outer_folds = random_int_seed_outer_folds
        self.random_int_seed_inner_folds = random_int_seed_inner_folds

    def read_folds(self, fold_indicator: np.ndarray):
        """Read a new folds specification. The user must make sure that later data is added according to these folds.

        Parameters
        ----------

        """
        if not isinstance(fold_indicator, np.ndarray):
            raise ValueError("Folds must be passed as numpy array!")

        if self.n_instances != len(fold_indicator):
            raise ValueError("Fold indicator contains less samples than the dataset.")

        if fold_indicator.min() != 0:
            raise ValueError("Folds numbers must start from 0. We are all computer scientists here...")

        if fold_indicator.max() != (len(np.unique(fold_indicator)) - 1):
            raise ValueError("Fold values are wrong. Highest fold unequal to unique values - 1")

        if len(self.predictors) > 0:
            raise ValueError("Chaining Folds even though predictors have been added based on old folds."
                             " Remove the predictor first or pass the folds earlier.")

        self.folds = fold_indicator

    # - Metatask to files
    def to_files(self, output_dir: str = ""):
        """Store the metatask in two files. One .csv and .json file

        The .csv file stores the complete meta-dataset. The .json stores additional and required metadata

        Parameters
        ----------
        output_dir:  str
            Directory in which the .json and .csv files for the metatask shall be stored.
        """
        # -- Store the predictions together with the dataset in one file
        file_path_csv = os.path.join(output_dir, "metatask_{}.csv".format(self.openml_task_id))
        self.meta_dataset.to_csv(file_path_csv, sep=",", header=True, index=False)

        # -- Store Meta data in its onw file
        file_path_json = os.path.join(output_dir, "metatask_{}.json".format(self.openml_task_id))

        with open(file_path_json, 'w', encoding='utf-8') as f:
            meta_data = self.meta_data

            # make some objects serializable
            meta_data["folds"] = meta_data["folds"].tolist()
            meta_data["validation_indices"] = {k: v.tolist() for k, v in meta_data["validation_indices"].items()}

            json.dump(meta_data, f, ensure_ascii=False, indent=4)

    def to_sharable_prediction_data(self, output_dir: str = ""):
        """Store Metatasks without dataset (e.g., all data but self.dataset's rows)"""

        # -- Remove all sensitive data from the metatask
        # Remove unsharable datasets
        self.dataset = self.dataset[0:0]

        # -- Store the metatasks files
        self.to_files(output_dir)

    # - Metatask from files
    def read_metatask_from_files(self, input_dir: str, openml_task_id: int, read_wo_dataset: bool = False):
        """ Build a metatask using data from files

        Parameters
        ----------
        input_dir : str
            Directory in which the .json and .csv files for the metatask are stored.
        openml_task_id: int
            The ID of the metatask/openml task that shall be read from the files
        read_wo_dataset: bool, default=False
            If the function is used to only read the meta data and prediction data from the files.
            Needed to determine which sanity checks to run.
        """
        # -- Read data
        file_path_csv = os.path.join(input_dir, "metatask_{}.csv".format(openml_task_id))
        file_path_json = os.path.join(input_dir, "metatask_{}.json".format(openml_task_id))

        # - Read meta data
        with open(file_path_json) as json_file:
            meta_data = json.load(json_file)

        # -- Init Meta Data
        for md_k in meta_data.keys():
            setattr(self, md_k, meta_data[md_k])

        # Post process special cases (setting arrays to arrays)
        self.folds = np.array(self.folds)
        self.validation_indices = {int(k): np.array(v) for k, v in self.validation_indices.items()}
        self.is_classification = "classification" == meta_data["task_type"]
        self.is_regression = "regression" == meta_data["task_type"]
        self.missing_metadata_in_file = [x for x in self.meta_data_keys if x not in meta_data.keys()]
        del meta_data

        # -- Read Dataset
        # Read cat columns correctly
        cat_columns = self.cat_feature_names[:]
        cat_labels = []

        # Add labels and predictors to cat columns for classification
        if self.is_classification:
            cat_labels.extend(self.predictors)  # predictions should be cat
            cat_labels.append(self.target_name)  # target is cat
            cat_labels.extend(self.validation_predictions_columns)  # val predictions
            cat_columns.extend(cat_labels)

        to_read_as_cat = {col_name: 'category' for col_name in cat_columns}
        meta_dataset = pd.read_csv(file_path_csv, dtype=to_read_as_cat)

        # Post process categories (will be skipped if not classification)
        if cat_labels:
            if any(not isinstance(x, str) for x in self.class_labels):
                raise ValueError("Something went wrong, class labels should be strings but are integers!")
            meta_dataset[cat_labels] = meta_dataset[cat_labels].apply(lambda x: x.cat.set_categories(self.class_labels),
                                                                      axis=0)

        # -- Init Datasets
        self.dataset = meta_dataset[self.feature_names + [self.target_name]]
        self.predictions_and_confidences = meta_dataset[self.pred_and_conf_cols]

        if self.validation_predictions_columns or self.validation_confidences_columns:
            self.validation_predictions_and_confidences = meta_dataset[self.validation_predictions_columns +
                                                                       self.validation_confidences_columns]
        else:
            self.validation_predictions_and_confidences = pd.DataFrame()
        del meta_dataset

        if not read_wo_dataset:
            self._dataset_sanity_checks()

    def from_sharable_prediction_data(self, input_dir: str, openml_task_id: int, dataset: pd.DataFrame):
        # Get Shared Data
        self.read_metatask_from_files(input_dir, openml_task_id, read_wo_dataset=True)

        # Re-order dataset to be identical to existing structure
        dataset = dataset[self.dataset.columns]

        # Fill dataset
        self.dataset.iloc[:] = dataset.iloc[:]

        # Solve Ground truth problems / formatting
        self._check_and_init_ground_truth()

        self._dataset_sanity_checks()

    # -- Metatask sanity checks code
    def _dataset_sanity_checks(self):
        # Simple Dataset Sanity Checks
        if list(self.dataset) != self.feature_names + [self.target_name]:
            raise ValueError("Columns formatting went wrong somehow.")

        if self.dataset[self.target_name].isnull().values.any():
            raise ValueError("Target Column contains nan values.")

        if len(self.dataset) != len(self.folds):
            raise ValueError("Folds are not equal to dataset size!")

        if self.is_classification:
            if not pd.api.types.is_categorical_dtype(self.dataset[self.target_name]):
                raise ValueError("For some reason, the target column is not categorical!")

    # --- Code for Post Processing a Metatask after it was build
    def remove_predictors(self, predictor_names):
        rel_conf_and_pred_cols = self.get_pred_and_conf_cols(predictor_names)
        only_conf_cols = [ele for ele in rel_conf_and_pred_cols if ele not in predictor_names]

        # Remove predictors from all parts of the metatask object
        self.predictions_and_confidences = self.predictions_and_confidences.drop(columns=rel_conf_and_pred_cols)
        self.predictors = [pred for pred in self.predictors if pred not in predictor_names]
        self.confidences = [ele for ele in self.confidences if ele not in only_conf_cols]
        self.bad_predictors = [pred for pred in self.bad_predictors if pred not in predictor_names]
        for pred_name in predictor_names:
            # save delete, does not raise key error if not in dict if we use pop here
            self.predictor_descriptions.pop(pred_name, None)
            self.predictor_corruptions_details.pop(pred_name, None)

        # Validation Data
        if self.use_validation_data:
            val_pred_cols = self.get_validation_predictions_columns(predictor_names)
            val_conf_cols = self.get_validation_confidences_columns(predictor_names)
            self.validation_predictions_and_confidences = self.validation_predictions_and_confidences.drop(
                columns=val_pred_cols + val_conf_cols)
            # If last validation data base model was removed act on it
            if not self.validation_predictions_columns:
                self.use_validation_data = False

        # Remove fold predictors afterwards because otherwise the validation data removal would not work
        self.fold_predictors = [pred for pred in self.fold_predictors if pred not in predictor_names]

    def filter_predictors(self, remove_bad_predictors: bool = True, remove_constant_predictors: bool = False,
                          remove_worse_than_random_predictors: bool = False,
                          score_metric: Optional[Callable] = None, maximize_metric: Optional[bool] = None,
                          max_number_predictors: Optional[int] = None):
        """A method to filter/removes predictors (base models) of a metatask.

        Parameters
        ----------
        remove_bad_predictors: bool, default=True
            Remove predictors that were deemed to be bad during the crawling process (because of errors in
            the prediction data).
        remove_constant_predictors: bool, default=False
            Remove constant predictors (base models that only predict 1 class for all instances)
        remove_worse_than_random_predictors: bool, default=False
            Remove predictions that are worse than a random predictor.
            Requires score_metric and maximize_metric to be not None.
        score_metric : metric function, default=None
            The metric function used to determine if a predictors is worse than a random predictor.
            Special format required due to OpenML's metrics.
        maximize_metric : bool, default=None
            Whether the metric computed by the metric function passed by score_metric is to be maximized or not.
        max_number_predictors: int, default=None
            If not none, keep only as many predictors as max_number_predictors. Discard predictors based on score_metric.
            Keep only the best predictors. Requires score_metric and maximize_metric to be not None.
        """

        if remove_bad_predictors:
            self.remove_predictors(self.bad_predictors)

        if remove_constant_predictors:
            # Get unique counts
            uc = self.meta_dataset[self.predictors].nunique()
            # Get constant predictors
            constant_predictors = uc[uc == 1].index.tolist()
            self.remove_predictors(constant_predictors)

        # In this case, for some fold, the number of predictors is too large.
        keep_only_a_subst_of_predictors = (max_number_predictors is not None) and any(
            max_number_predictors < n_per_fold for n_per_fold in self.n_predictors_per_fold)

        if (keep_only_a_subst_of_predictors or remove_worse_than_random_predictors) and self.predictors:
            if any(x is None for x in [score_metric, maximize_metric]):
                raise ValueError("We require a metric and whether the metric is to be maximized! "
                                 "But one of them is none.")

            # FIXME: if fold predictors and none-fold predictors are mixed in a metatask, this could remove a
            #  non-fold predictor in a later fold that is required to have max_number_predictors in an earlier
            #  fold. Let us assume for now that we only have none-fold predictors or fold predictors but no mix.

            # FIXME: currently only evaluate this for the first fold if we have only none-fold predictors.

            # Check if pure predictors
            if self.fold_predictors and (self.fold_predictors != self.predictors):
                raise NotImplementedError("Not just Fold Predictor or None-Fold Predictors but mixture of both!")

            predictors_to_remove = set()

            # Iterative over folds and handle each fold individually
            for idx, train_metadata, test_metadata in self.fold_split(return_fold_index=True):
                fold_predictors_idx_to_remove = []

                # -- Get Validation Data from Metatask which is used for filter
                if self.use_validation_data:
                    _, bm_y_true, _, _, bm_Y_pred, _ = self.split_meta_dataset(train_metadata, fold_idx=idx)
                    new_cols = [self.from_validation_predictor_name_to_predictor_name(x) for x in bm_Y_pred.columns]
                    bm_Y_pred.columns = new_cols
                else:
                    _, y_test, test_base_predictions, _, _, _ = self.split_meta_dataset(test_metadata, fold_idx=idx)

                    # Split to get validation data
                    bm_y_true, _, bm_Y_pred, _ = train_test_split(y_test, test_base_predictions, test_size=0.5,
                                                                  random_state=0, stratify=y_test)

                # -- Get Base Model Performance (we ignore confidences for this comparison!)
                bm_performances = bm_Y_pred.apply(lambda x: score_metric(bm_y_true, x), axis=0)

                # -- Get list of worse than random predictors
                if remove_worse_than_random_predictors:
                    # Get random performance
                    random_predictions = np.random.choice(self.class_labels, bm_y_true.shape[0])
                    random_performance = score_metric(bm_y_true, random_predictions)

                    # Get worse than random predictors based on metric (max/min)
                    if maximize_metric:
                        wtr_idx = bm_performances < random_performance
                    else:
                        wtr_idx = bm_performances > random_performance

                    # Save which predictors to remove
                    fold_predictors_idx_to_remove.extend(np.flatnonzero(wtr_idx).tolist())
                    fold_predictors_idx_to_remove = list(set(fold_predictors_idx_to_remove))

                # -- Get list of not in top k predictors
                if keep_only_a_subst_of_predictors:
                    mask_already_removed = np.setdiff1d(np.arange(bm_performances.shape[0]),
                                                        fold_predictors_idx_to_remove)
                    tmp_idx = np.arange(bm_performances.shape[0])[mask_already_removed]
                    tmp_bm_performance = np.array(bm_performances)[mask_already_removed]

                    # -- Evaluate if previous steps have removed enough predictors already.
                    if len(tmp_bm_performance) >= max_number_predictors:
                        # Still too many predictors

                        if maximize_metric:
                            tmp_bm_performance = -tmp_bm_performance

                        not_in_top_set = np.argsort(tmp_bm_performance)[max_number_predictors:].tolist()
                        fold_predictors_idx_to_remove.extend(tmp_idx[not_in_top_set].tolist())
                        fold_predictors_idx_to_remove = list(set(fold_predictors_idx_to_remove))

                predictors_to_remove.update(bm_performances[fold_predictors_idx_to_remove].index.tolist())

                # If you only have predictors for all folds, break after first fold
                # otherwise, it could remove much more predictors
                if not self.fold_predictors:
                    break

            self.remove_predictors(list(predictors_to_remove))

    # -- Run/Benchmark Functions
    def get_indices_for_fold(self, fold_idx, return_indices=False):
        train_indices = self.folds != fold_idx
        test_indices = self.folds == fold_idx

        if return_indices:
            return np.where(train_indices)[0], np.where(test_indices)[0]

        return train_indices, test_indices

    def fold_split(self, return_fold_index=False):
        # Return split copy of metadataset
        for i in range(self.max_fold + 1):
            train_indices, test_indices = self.get_indices_for_fold(i)

            if return_fold_index:
                yield i, self.meta_dataset.iloc[train_indices].copy(), self.meta_dataset.iloc[test_indices].copy()
            else:
                yield self.meta_dataset.iloc[train_indices].copy(), self.meta_dataset.iloc[test_indices].copy()

    def split_meta_dataset(self, meta_dataset, fold_idx: Optional[int] = None) -> Tuple[
        pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits the meta dataset into its subcomponents

        Parameters
        ----------
        meta_dataset: self.meta_dataset
        fold_idx: int, default=None
            If int, the int is used to filter fold related data such that only the data for the fold with fold_idx
            remains in the returned data.

        Returns
        -------
        features : pd.DataFrame
            The features of the original dataset
        ground_truth: pd.Series
            The ground_truth of the original dataset
        predictions: pd.DataFrame
            The predictions of the base models
        confidences: pd.DataFrame
            The confidences of the base models
        validation_predictions: pd.DataFrame
            The predictions of the base models on the validation of a fold
        validation_confidences: pd.DataFrame
            The confidences of the base models on the validation of a fold
        """

        if fold_idx is None:
            predictions_columns = self.predictors
            confidences_columns = self.confidences

            validation_predictions_columns = self.validation_predictions_columns
            validation_confidences_columns = self.validation_confidences_columns

        else:
            if (fold_idx < 0) or fold_idx > self.max_fold:
                raise ValueError("Fold index passed is not equal to an existing fold.")

            # Handle Validation data
            validation_predictions_columns = [col_n for col_n in self.validation_predictions_columns
                                              if col_n.endswith(self.to_validation_predictor_name("", fold_idx))]
            validation_confidences_columns = [col_n for col_n in self.validation_confidences_columns
                                              if col_n.endswith(self.to_validation_predictor_name("", fold_idx))]

            # Handle Fold Predictors - add all normal predictors and fold predictors if the fold is correct
            predictions_columns = self.get_predictors_for_fold(fold_idx)

            confidences_columns = self.get_conf_cols(predictions_columns)

        return meta_dataset[self.feature_names], meta_dataset[self.target_name], meta_dataset[predictions_columns], \
               meta_dataset[confidences_columns], meta_dataset[validation_predictions_columns], \
               meta_dataset[validation_confidences_columns]

    @staticmethod
    def _save_fold_results(y_true, y_pred, fold_idx, out_path, technique_name, classification=True):
        """Code to Save Fold results in a file such that we can add any other folds results in the future"""
        if out_path is None:
            return

        # Path test
        path_exists = os.path.exists(out_path)

        # Get data that is to be saved
        index_metatask = y_true.index
        fold_indicator = [fold_idx for _ in range(len(y_true))]
        res_df = pd.DataFrame(np.array([y_true, index_metatask, fold_indicator, y_pred]).T,
                              columns=["ground_truth", "Index-Metatask", "Fold", technique_name])

        # Keep type correct
        if classification:
            # Note: while we tell read_csv that technique name should be read as string, it does not matter if
            #   technique name is not in the file to be read. Hence it works for loading it initially
            input_dtype = "string"
            res_df = res_df.astype(input_dtype).astype({"Index-Metatask": int, "Fold": int})
        else:
            input_dtype = None

        # Save folds base
        if fold_idx == 0:
            if path_exists:
                # Load old data
                tmp_df = pd.read_csv(out_path, dtype=input_dtype).astype(
                    {"Index-Metatask": int, "Fold": int}).set_index(["ground_truth", "Index-Metatask", "Fold"])
                # Pre-fill technique values
                tmp_df[technique_name] = pd.Series(np.nan, index=np.arange(len(tmp_df)))
                # Insert new values into data
                res_df = res_df.set_index(["ground_truth", "Index-Metatask", "Fold"])

                # Try-catch to detect (potential) random seed error
                try:
                    tmp_df.loc[res_df.index, technique_name] = res_df
                except KeyError:
                    raise KeyError("Wrong fold/index/ground-truth combinations found in results file." +
                                   " Did you create the existing results file with a different random seed for" +
                                   " the meta train test split? " +
                                   " To fix this delete the existing files or adjust the random seed.")

                tmp_df.reset_index().sort_values(by=["Index-Metatask"]).to_csv(out_path, index=False)
            else:
                res_df.to_csv(out_path, index=False)

        else:
            # Check error
            if not path_exists:
                raise RuntimeError("Somehow a later fold is trying to be saved first. " +
                                   "Did the output file got deleted between folds?")

            # Read so-far data
            tmp_df = pd.read_csv(out_path, dtype=input_dtype).astype(
                {"Index-Metatask": int, "Fold": int}).set_index(["ground_truth", "Index-Metatask", "Fold"])

            if len(list(tmp_df)) == 1:
                # Initial case, first time file building
                res_df.to_csv(out_path, mode='a', header=False, index=False)
            else:
                # Default case later, same as above wihtout init
                res_df = res_df.set_index(["ground_truth", "Index-Metatask", "Fold"])
                tmp_df.loc[res_df.index, technique_name] = res_df
                tmp_df.reset_index().sort_values(by=["Index-Metatask"]).to_csv(out_path, index=False)

    def run_ensemble_on_all_folds(self, technique, technique_args: dict, technique_name,
                                  use_validation_data_to_train_ensemble_techniques: bool = False,
                                  meta_train_test_split_fraction: float = 0.5, meta_train_test_split_random_state=0,
                                  pre_fit_base_models: bool = False, base_models_with_names: bool = False,
                                  label_encoder=False, preprocessor=None, output_file_path=None, oracle=False,
                                  probability_calibration="no", return_scores: Optional[Callable] = None):
        """Run an ensemble technique on all folds and return the results

        The current implementation builds fake base models by default such that we can evaluate methods this way.

        !Warning!: we will overwrite older data if the technique name is already existing in the results file!
        We will also not delete an existing file. If a file exist, we only add to/overwrite the file.

        Parameters
        ----------
        technique: sklearn estimator like object
            The Ensemble Technique (as a sklearn like fit/predict style) model.
        technique_args: dict
            The arguments that shall be supplied to the ensemble
        technique_name: str
            Name of the technique used to identify the technique later on in the saved results.
        use_validation_data_to_train_ensemble_techniques: bool, default=False
            Whether to use validation data to train the ensemble techniques (through the faked base models) and use the
            fold's prediction data to evaluate the ensemble technique.
            If True, the metataks requires validation data. If False, test predictions of a fold are split into
            meta_train and meta_test subsets where the meta_train subset is used to train the ensemble techniques and
            meta_test to evaluate the techniques.
        meta_train_test_split_fraction: float, default=0.5
            The fraction for the meta train/test split. Only used if
        meta_train_test_split_random_state: int, default=0
            The randomness for the meta train/test split.
        pre_fit_base_models: bool, default=False
            Whether or not the base models need to be fitted to be passed to the ensemble technique.
        base_models_with_names: bool, default=False
            Whether or not the base models' list should contain the model and its name.
        label_encoder: bool, default=False
            Whether the ensemble technique expects that a label encoder is applied to the fake models. Often required
            for sklearn ensemble techniques.
        preprocessor: sklearn-like transformer, default=None
            Function used to preprocess the data for later. called fit_transform on X_train and transform on X_test.
        output_file_path: str, default=None
            File path where the results of the folds shall be stored. If none, we do not store anything.
            We assume the file is in the correct format if it exists and will create it if it does not exit.
            Here, no option to purge/delete existing files is given. This is must be done in an outer scope.
        oracle: bool, default=False
            Whether the ensemble technique is an oracle. If true, we pass and call the method differently.
        probability_calibration: {"sigmoid", "isotonic", "auto", "no"}, default="no"
            What type of probability calibration (see https://scikit-learn.org/stable/modules/calibration.html)
            shall be applied to the base models:

                - "sigmoid": Use CalibratedClassifierCV with method="sigmoid"
                - "isotonic": Use CalibratedClassifierCV with method="isotonic"
                - "auto": Determine which method to use for CalibratedClassifierCV depending on the number of instances.
                - "no": Do not use probability calibration.

            If pre_fit_base_models is False, CalibratedClassifierCV is employed with ensemble="False" to simulate
            cross_val_predictions by our Faked Base Models.
            If pre_fit_base_models is True, CalibratedClassifierCV is employed with cv="prefit" beforehand such that
            we "replace" the base models with calibrated base models.
        return_scores: Callable, default=None
            If the evaluation shall return the scores for each fold. If not None, a metric function is expected.
        """
        # TODO -- Parameter Preprocessing / Checking
        #   Add safety check for file path here or something
        #   Check if probability_calibration has correct string names
        #   Check metric / scorer object
        if use_validation_data_to_train_ensemble_techniques and (not self.use_validation_data):
            raise ValueError("Metatask has no validation data but use_validation_to_train_ensemble_techniques is True.")

        # -- Re-names
        val_data = use_validation_data_to_train_ensemble_techniques
        m_frac = meta_train_test_split_fraction
        m_rand = meta_train_test_split_random_state

        # -- Iterate over Folds
        fold_scores = []
        for idx, train_metadata, test_metadata in self.fold_split(return_fold_index=True):

            # -- Get Data from Metatask
            X_train, y_train, _, _, val_base_predictions, val_base_confidences = self.split_meta_dataset(train_metadata,
                                                                                                         fold_idx=idx)
            X_test, y_test, test_base_predictions, test_base_confidences, _, _ = self.split_meta_dataset(test_metadata,
                                                                                                         fold_idx=idx)

            # -- Employ Preprocessing and input validation
            train_indices = X_train.index.to_numpy()
            X_train, X_test, y_train, y_test = self._check_fold_data_for_ensemble(X_train, X_test, y_train, y_test,
                                                                                  preprocessor)
            # -- Get Data for Fake Base Model and Evaluation
            # - Data on which the base models have been trained and had to predict for the fold
            test_base_model_train_X = X_train
            test_base_model_train_y = y_train

            # - Validation Data
            if not val_data:
                ensemble_train_X, ensemble_test_X, ensemble_train_y, ensemble_test_y, fake_base_model_known_X, \
                fake_base_model_known_predictions, fake_base_model_known_confidences, val_base_model_train_X, \
                val_base_model_train_y = self._get_data_for_ensemble_without_validation(X_train, X_test, y_train,
                                                                                        y_test,
                                                                                        test_base_predictions,
                                                                                        test_base_confidences,
                                                                                        m_frac, m_rand)
            else:
                # Get iloc indices of validation data (Default value is for backwards compatibility)
                fold_validation_indices = self.validation_indices.get(idx, train_indices)
                # iloc_indices corresponds to np.arange(len(X_test)) for cross-validation
                iloc_indices = np.intersect1d(fold_validation_indices, train_indices, return_indices=True)[2]

                # Get Data
                ensemble_train_X, ensemble_test_X, ensemble_train_y, ensemble_test_y, fake_base_model_known_X, \
                fake_base_model_known_predictions, fake_base_model_known_confidences, val_base_model_train_X, \
                val_base_model_train_y = self._get_data_for_ensemble_with_validation(X_train[iloc_indices], X_test,
                                                                                     y_train[iloc_indices], y_test, idx,
                                                                                     val_base_predictions.iloc[
                                                                                         iloc_indices],
                                                                                     val_base_confidences.iloc[
                                                                                         iloc_indices],
                                                                                     test_base_predictions,
                                                                                     test_base_confidences)

            # -- Build Fake Base Models
            base_models = self._build_fake_base_models(test_base_model_train_X, test_base_model_train_y,
                                                       val_base_model_train_X, val_base_model_train_y,
                                                       ensemble_train_X, ensemble_train_y,
                                                       fake_base_model_known_X,
                                                       fake_base_model_known_predictions,
                                                       fake_base_model_known_confidences,
                                                       pre_fit_base_models, base_models_with_names,
                                                       label_encoder, probability_calibration)

            # -- Run Ensemble for Fold
            y_pred_ensemble_model = self._run_ensemble_on_data(base_models, technique, technique_args,
                                                               ensemble_train_X, ensemble_test_X, ensemble_train_y,
                                                               ensemble_test_y, oracle)

            # -- Post Process Results
            self._save_fold_results(ensemble_test_y, y_pred_ensemble_model, idx, output_file_path, technique_name)

            # -- Save scores for return
            if return_scores is not None:
                fold_scores.append(return_scores(ensemble_test_y, y_pred_ensemble_model))

        # -- Return Score
        if return_scores is not None:
            return fold_scores

    def _build_fake_base_models(self, test_base_model_train_X, test_base_model_train_y,
                                val_base_model_train_X, val_base_model_train_y,
                                ensemble_train_X, ensemble_train_y, fake_base_model_known_X,
                                fake_base_model_known_predictions, fake_base_model_known_confidences,
                                pre_fit_base_models, base_models_with_names, label_encoder, probability_calibration):
        """Build fake base models from data.

        New Parameters (rest can be found in signature of run_ensemble_on_all_folds)
        ----------
        test_base_model_train_X: array-like
            The instances on which the base models are trained for the (outer) evaluation
        test_base_model_train_y: array-like
            The labels on which the base models are trained for the (outer) evaluation
        val_base_model_train_X:
            The instances on which the base models are trained for the validation data.
        val_base_model_train_y: array-like
            The labels on which the base models are trained for the validation data.
        ensemble_train_X: array-like
            The instances on which the ensemble technique is to be trained.
        ensemble_train_y: array-like
            The labels on which the ensemble technique is to be trained.
        fake_base_model_known_X: array-like
            The instances from the test or training data for which the base model must know predictions.
        fake_base_model_known_predictions: pd.DataFrame
            Predictions for the instances of fake_base_model_known_X
        fake_base_model_known_confidences: pd.DataFrame
            Confidences for the instances of fake_base_model_known_X
        """

        # -- Build ensemble technique
        base_models = _initialize_fake_models(test_base_model_train_X, test_base_model_train_y,
                                              val_base_model_train_X, val_base_model_train_y, fake_base_model_known_X,
                                              fake_base_model_known_predictions, fake_base_model_known_confidences,
                                              pre_fit_base_models, base_models_with_names, label_encoder,
                                              self.to_confidence_name)

        # -- Probability Calibration
        base_models = probability_calibration_for_faked_models(base_models, ensemble_train_X, ensemble_train_y,
                                                               probability_calibration, pre_fit_base_models)

        return base_models

    @staticmethod
    def _run_ensemble_on_data(base_models, technique, technique_args, ensemble_train_X, ensemble_test_X,
                              ensemble_train_y, ensemble_test_y, oracle):
        """Run an ensemble technique for the given data and return its predictions

        New Parameters (rest can be found in signature of run_ensemble_on_all_folds or _build_fake_base_models)
        ----------
        base_models: List of fake base models
            A list of base models that shall be used by the ensemble technique.
        ensemble_test_X: array-like
            The instances on which the ensemble technique shall predict on.
        ensemble_test_y: array-like
            The labels which the ensemble technique shall predict (only used for oracle-like techniques).
        """

        # -- Build ensemble model
        ensemble_model = technique(base_models, **technique_args)

        # -- Fit and Predict
        ensemble_model.fit(ensemble_train_X, ensemble_train_y)

        if oracle:
            # Special case for virtual oracle-like predictors
            y_pred_ensemble_model = ensemble_model.oracle_predict(ensemble_test_X, ensemble_test_y)
        else:
            y_pred_ensemble_model = ensemble_model.predict(ensemble_test_X)

        return y_pred_ensemble_model

    @staticmethod
    def _check_fold_data_for_ensemble(X_train, X_test, y_train, y_test, preprocessor):
        if preprocessor is not None:
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)
        else:
            # make sure X_train and X_test are arrays from here onwards
            X_train = X_train.to_numpy()
            X_test = X_test.to_numpy()

        # Make sure y_train and y_test are arrays
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()

        return X_train, X_test, y_train, y_test

    # -- Data for Ensemble Getters
    @staticmethod
    def _get_data_for_ensemble_without_validation(X_train, X_test, y_train, y_test, test_base_predictions,
                                                  test_base_confidences, meta_train_test_split_fraction,
                                                  meta_train_test_split_random_state):
        """Split for ensemble technique evaluation only on fold predictions to get trainings data for the ensemble
        """

        # -- In this case, the base models have been fitted on the same data for the validation and test data
        #   (because we take a subset of the test data as validation data)
        val_base_model_train_X = X_train
        val_base_model_train_y = y_train

        # -- Data on the predictions of the original base model
        fake_base_model_known_X = X_test
        fake_base_model_known_predictions = test_base_predictions
        fake_base_model_known_confidences = test_base_confidences

        # Split of the original test data corresponding to the parts of the predictions used for train and test
        ensemble_train_X, ensemble_test_X, ensemble_train_y, ensemble_test_y \
            = train_test_split(X_test, y_test, test_size=meta_train_test_split_fraction,
                               random_state=meta_train_test_split_random_state, stratify=y_test)

        return ensemble_train_X, ensemble_test_X, ensemble_train_y, ensemble_test_y, fake_base_model_known_X, \
               fake_base_model_known_predictions, fake_base_model_known_confidences, val_base_model_train_X, \
               val_base_model_train_y

    def _get_data_for_ensemble_with_validation(self, X_train, X_test, y_train, y_test, fold_idx,
                                               val_base_predictions, val_base_confidences,
                                               test_base_predictions, test_base_confidences):
        """Build data for validation based evaluation

        Fake Model must be able to predict for all instances of the validation data and the fold's predictions.
        Hence, it must get all the data needed for that.
        """

        # -- Select the relevant subset of training instances that are part of the validation data
        val_base_model_train_X = X_train
        val_base_model_train_y = y_train

        # -- Get known/oracle data for fake base model
        # Therefore, combine validation and test instances
        fake_base_model_known_X = np.vstack((val_base_model_train_X, X_test))

        # Also need to combine predictions / confidences (keep as DF for columns)
        # Have to rename columns to use DFs with validation data, here remove fold postfix to achieve this
        p_rename = {self.to_validation_predictor_name(col_name, fold_idx): col_name for col_name
                    in list(test_base_predictions)}
        c_rename = {self.to_validation_predictor_name(col_name, fold_idx): col_name for col_name
                    in list(test_base_confidences)}
        fake_base_model_known_predictions = pd.concat([val_base_predictions.rename(columns=p_rename),
                                                       test_base_predictions], axis=0)
        fake_base_model_known_confidences = pd.concat([val_base_confidences.rename(columns=c_rename),
                                                       test_base_confidences], axis=0)

        # Finally set which subsets are used for what
        # (These correspond to the subsets passed to the base model to get known predictions/confidences)
        ensemble_train_X = val_base_model_train_X
        ensemble_train_y = val_base_model_train_y
        ensemble_test_X = X_test
        ensemble_test_y = y_test

        # Return
        return ensemble_train_X, ensemble_test_X, ensemble_train_y, ensemble_test_y, fake_base_model_known_X, \
               fake_base_model_known_predictions, fake_base_model_known_confidences, val_base_model_train_X, \
               val_base_model_train_y

    # ---- Experimental Functions
    def _exp_yield_evaluation_data_across_folds(self, meta_train_test_split_fraction,
                                                meta_train_test_split_random_state,
                                                pre_fit_base_models, base_models_with_names, label_encoder,
                                                preprocessor, include_test_data=False):
        # FIXME, extend to validation data usage?
        for idx, train_metadata, test_metadata in self.fold_split(return_fold_index=True):
            X_train, y_train, _, _, _, _ = self.split_meta_dataset(train_metadata)
            X_test, y_test, test_base_predictions, test_base_confidences, _, _ = self.split_meta_dataset(test_metadata)

            if preprocessor is not None:
                X_train = preprocessor.fit_transform(X_train)
                X_test = preprocessor.transform(X_test)

            X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(X_test, y_test,
                                                                                    test_size=meta_train_test_split_fraction,
                                                                                    random_state=meta_train_test_split_random_state,
                                                                                    stratify=y_test)

            base_models = _initialize_fake_models(X_train, y_train, X_train, y_train, X_test, test_base_predictions,
                                                  test_base_confidences, pre_fit_base_models, base_models_with_names,
                                                  label_encoder, self.to_confidence_name)

            if include_test_data:
                assert_meta_train_pred, assert_meta_test_pred, assert_meta_train_conf, \
                assert_meta_test_conf = train_test_split(test_base_predictions, test_base_confidences,
                                                         test_size=meta_train_test_split_fraction,
                                                         random_state=meta_train_test_split_random_state,
                                                         stratify=y_test)
                yield base_models, X_meta_train, X_meta_test, y_meta_train, y_meta_test, assert_meta_train_pred, \
                      assert_meta_test_pred, assert_meta_train_conf, assert_meta_test_conf, X_train, y_train
            else:
                yield base_models, X_meta_train, X_meta_test, y_meta_train, y_meta_test

    def _exp_yield_data_for_base_model_across_folds(self, folds_to_run: Optional[List[int]] = None):
        """ Get Folds Data to Train Base Models

        Parameters
        ----------
        folds_to_run: List of int, default=None
            If None, nothing changes.
            If not None, the function will only return the fold data for the fold indices
            specified in the list.
        """

        # Only used for tests currently
        for idx, train_metadata, test_metadata in self.fold_split(return_fold_index=True):
            # -- Check if Fold is wanted, if not skip
            if (folds_to_run is not None) and (idx not in folds_to_run):
                continue

            # -- Get Data from Metatask
            X_train, y_train, _, _, _, _ = self.split_meta_dataset(train_metadata)
            X_test, y_test, _, _, _, _ = self.split_meta_dataset(test_metadata)

            yield idx, X_train, X_test, y_train, y_test
