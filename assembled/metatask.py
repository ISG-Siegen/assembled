import pandas as pd
import numpy as np
import os
import json

from assembled.compatibility.faked_classifier import probability_calibration_for_faked_models, initialize_fake_models
from typing import List, Tuple, Optional, Callable, Union
from sklearn.model_selection import train_test_split


class MetaTask:
    def __init__(self):
        """Metatask, a meta version of a normal OpenML Task

        The Metatask contains the predictions and confidences (e.g. sklearn's predict_proba) of specific base models
        and the data of the original (OpenML) task. Moreover, additional side information are captured.
        This object is filled via functions and thus it is empty initially.
        """

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
        self.predictor_corruptions_details = {}
        self.confidences = []  # List of column names of confidences
        self.validation_predictions_and_confidences = pd.DataFrame()
        self.use_validation_data = False

        # -- Selection constrains (used to find the data of this metatask)
        self.selection_constraints = {}

        # -- Other
        self.supported_task_types = {"classification", "regression"}
        self.confidence_prefix = "confidence."
        self.fold_postfix = ".fold"

        # -- Randomness
        self.random_seed_for_outer_folds_split = None
        self.randomness_base_seed_validation_splits = None

        # -- Backwards Compatibility
        self.missing_metadata_in_file = None

    @property
    def n_classes(self):
        return len(self.class_labels)

    @property
    def n_instances(self):
        return len(self.dataset)

    @property
    def meta_dataset(self):
        return pd.concat([self.dataset, self.predictions_and_confidences, self.validation_predictions_and_confidences],
                         axis=1)

    @property
    def meta_data_keys(self):
        return ["openml_task_id", "dataset_name", "target_name", "class_labels", "predictors", "confidences",
                "predictor_descriptions", "bad_predictors", "confidence_prefix", "feature_names",
                "cat_feature_names", "selection_constraints", "task_type", "predictor_corruptions_details",
                "use_validation_data", "random_seed_for_outer_folds_split", "randomness_base_seed_validation_splits",
                "folds"]

    @property
    def meta_data(self):
        meta_data = {md_k: getattr(self, md_k) for md_k in self.meta_data_keys}

        return meta_data

    @property
    def ground_truth(self) -> pd.Series:
        return self.dataset[self.target_name]

    @ground_truth.setter
    def ground_truth(self, value):
        self.dataset[self.target_name] = value

    @property
    def confs_per_class_per_predictor(self):
        """ Confidence-Class-Predictor Relationships

        A dict of dicts detailing for each predictor name the relationship between the class labels and the column
        names of confidences.
        """
        return {col_name: {n: "{}{}.{}".format(self.confidence_prefix, n, col_name) for n in self.class_labels}
                for col_name in self.predictors}

    @property
    def pred_and_conf_cols(self):
        # Predictor and confidence columns in a specific order 
        return [ele for slist in [[col_name, *["{}{}.{}".format(self.confidence_prefix, n, col_name)
                                               for n in self.class_labels]]
                                  for col_name in self.predictors] for ele in slist]

    def get_pred_and_conf_cols(self, predictor_names):
        # return relevant predictor and confidence columns in a specific order
        return [ele for slist in [[col_name, *["{}{}.{}".format(self.confidence_prefix, n, col_name)
                                               for n in self.class_labels]]
                                  for col_name in predictor_names] for ele in slist]

    @property
    def non_cat_feature_names(self):
        return [f for f in self.feature_names if f not in self.cat_feature_names]

    @property
    def max_fold(self):
        return int(self.folds.max())

    @property
    def validation_predictions_columns(self):
        if not self.use_validation_data:
            return []

        return [pred_col + "{}{}".format(self.fold_postfix, i) for pred_col in self.predictors for i in
                range(self.max_fold + 1)]

    @property
    def validation_confidences_columns(self):
        if not self.use_validation_data:
            return []

        return ["{}{}.{}".format(self.confidence_prefix, n, p_name) for n in
                self.class_labels for p_name in self.validation_predictions_columns]

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

        # -- Save Data
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.target_name = target_name
        self.class_labels = sorted(class_labels)
        self.feature_names = feature_names
        self.cat_feature_names = cat_feature_names
        self.openml_task_id = openml_task_id
        self.folds = folds_indicator

        # Handle task type
        self.task_type = task_type
        if task_type not in self.supported_task_types:
            raise ValueError("Unsupported Task Type Str '{}' was found.".format(task_type))
        self.is_classification = task_type == "classification"
        self.is_regression = task_type == "regression"

        self._check_and_init_ground_truth()

    def add_predictor(self, predictor_name: str, predictions: np.ndarray, confidences: Optional[np.ndarray] = None,
                      conf_class_labels: Optional[List[str]] = None, predictor_description: Optional[str] = None,
                      bad_predictor: bool = False, corruptions_details: Optional[dict] = None,
                      validation_data: Optional[List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = None):
        """Add a new predictor (base model) to the metatask

        Parameters
        ----------
        predictor_name: str
            name of the predictor; must be unique within a given metatask! (Overwriting is not supported yet)
        predictions: array-like, (n_samples,)
            Cross-val-predictions for the predictor that correspond to the fold_indicators of the metatask
        confidences: array-like, Optional, (n_samples, n_classes), default=None
            Confidences of the prediction. If None, due to Regression tasks or no confidences, use default confidences.
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
            The validation of the predictor for all folds.
            We assume an input of a list of lists which contain: List[the fold index, the predictions on the validation
            data, confidences on the validation data, indices of the validation data].
        """

        # -- Input Checks
        if conf_class_labels is not None:
            if not set(conf_class_labels) == set(self.class_labels):
                raise ValueError("Unknown class labels as input. Expected: {}; Got: {}".format(self.class_labels,
                                                                                               conf_class_labels))
        if predictor_name in self.predictors:
            raise ValueError("The name of the predictor already exist. Can not overwrite."
                             " The name must be unique. Got: {}".format(predictor_name))

            # -- Build Data for Dataframe

        # - Predictions to Series with correct name
        if isinstance(predictions, pd.Series):
            predictions = predictions.rename(predictor_name)
        else:
            # Assume it is an array-like, if not duck-typing will tell us
            predictions = pd.Series(predictions, name=predictor_name)

        # - Confidences to DataFrame
        if confidences is None:
            # TODO add code for regression and non-confidences data (fake confidences)
            raise NotImplementedError("We currently require confidences for a predictor! Sorry...")
        else:
            # Get names for conf columns
            class_labels_to_use = conf_class_labels if conf_class_labels is not None else self.class_labels
            conf_col_names = ["{}{}.{}".format(self.confidence_prefix, n, predictor_name) for n in class_labels_to_use]

            # Get DF with names
            if isinstance(confidences, pd.DataFrame):
                re_name = {confidences.columns[i]: n for i, n in enumerate(conf_col_names)}
                confidences = confidences.rename(re_name, axis=1)
            else:
                # Assume it is an array-like, if not duck-typing will tell us
                confidences = pd.DataFrame(confidences, columns=conf_col_names)

        # - Check description
        if predictor_description is None:
            raise NotImplementedError("Automatic Description building is not yet supported. "
                                      "Please set the description by hand.")

        # - Check, sort, and store the validation data
        if validation_data is not None:
            self.use_validation_data = True

            # Iterate over validation data and store
            for fold_idx, val_preds, val_confs, val_indices in validation_data:
                save_name = predictor_name + "{}{}".format(self.fold_postfix, fold_idx)

                # --- Val predictions
                if isinstance(val_preds, pd.Series):
                    val_pred_data = val_preds.rename(save_name)
                else:
                    # Assume it is an array-like, if not duck-typing will tell us
                    val_pred_data = pd.Series(val_preds, name=save_name)

                # --- Val confidences
                if val_confs is None:
                    raise NotImplementedError("We currently require validation confidences for a predictor! Sorry...")
                else:
                    class_labels_to_use = conf_class_labels if conf_class_labels is not None else self.class_labels
                    val_conf_col_names = ["{}{}.{}".format(self.confidence_prefix, n, save_name) for n in
                                          class_labels_to_use]

                    # Get DF with names
                    if isinstance(val_confs, pd.DataFrame):
                        re_name = {val_confs.columns[i]: n for i, n in enumerate(val_conf_col_names)}
                        val_confs = val_confs.rename(re_name, axis=1)
                    else:
                        # Assume it is an array-like, if not duck-typing will tell us
                        val_confs = pd.DataFrame(val_confs, columns=val_conf_col_names)

                    val_pred_data = pd.concat([val_pred_data, val_confs], axis=1)

                # --- Preprocess data to be stored
                remaining_indices = self.get_indices_for_fold(fold_idx, return_indices=True)[1]
                val_pred_data_filler = pd.DataFrame()
                for col in val_pred_data.columns:
                    val_pred_data_filler[col] = np.full(len(remaining_indices), np.nan)

                val_pred_data["tmp_idx"] = val_indices
                val_pred_data_filler["tmp_idx"] = remaining_indices

                val_pred_data = pd.concat([val_pred_data, val_pred_data_filler], axis=0)
                val_pred_data = val_pred_data.sort_values(by="tmp_idx").drop(columns=["tmp_idx"]).reset_index(drop=True)

                # --- Store Validation Data
                self.validation_predictions_and_confidences = pd.concat(
                    [self.validation_predictions_and_confidences, val_pred_data], axis=1)

                # -- Add prediction data to metatask
        else:
            if self.use_validation_data:
                raise ValueError("Validation data is required if at least one other base model has validation!")

        self.predictions_and_confidences = pd.concat([self.predictions_and_confidences, predictions, confidences],
                                                     axis=1)
        self.predictors.append(predictor_name)
        self.confidences.extend(conf_col_names)
        self.predictor_descriptions[predictor_name] = predictor_description

        if bad_predictor:
            self.bad_predictors.append(predictor_name)
        if corruptions_details is not None:
            self.predictor_corruptions_details[predictor_name] = corruptions_details

    def _check_and_init_ground_truth(self):
        # -- Process and Check ground truth depending on task type
        ground_truth = self.ground_truth

        if self.is_classification:
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
        else:
            # We have nothing to check for Regression // No checks implemented
            pass

        self.ground_truth = ground_truth

    def read_selection_constraints(self, selection_constraints):
        """Fill the constrains used to build the metatask.

        This only updates but does not overwrite existing keys.

        Parameters
        ----------
        selection_constraints : dict
            A dict containing the names and values for selection constraints
        """
        self.selection_constraints.update(selection_constraints)

    def read_randomness(self, random_seed_for_outer_folds_split: Union[int, str],
                        randomness_base_seed_validation_splits: Optional[int] = None):
        """

        Parameters
        ----------
        random_seed_for_outer_folds_split: int or str
            The random seed (integer) used to create the folds. If not available, pass a short description why.
        randomness_base_seed_validation_splits: int, default=None
            We assume that the splits used to get the validation data were generated by some controlled randomness.
            That is, a RandomState object initialized with a base seed was (re-)used each fold to generate the splits.
            Here, we want to have the base seed to store

        """
        self.random_seed_for_outer_folds_split = random_seed_for_outer_folds_split
        self.randomness_base_seed_validation_splits = randomness_base_seed_validation_splits

    def read_metatask_from_files(self, input_dir: str, openml_task_id: int):
        """ Build a metatask using data from files

        Parameters
        ----------
        input_dir : str
            Directory in which the .json and .csv files for the metatask are stored.
        openml_task_id: int
            The ID of the metatask/openml task that shall be read from the files
        """
        # -- Read data
        file_path_csv = os.path.join(input_dir, "metatask_{}.csv".format(openml_task_id))
        file_path_json = os.path.join(input_dir, "metatask_{}.json".format(openml_task_id))

        # - Read meta data
        with open(file_path_json) as json_file:
            meta_data = json.load(json_file)

        # - Read cat columns correctly
        cat_f = meta_data["cat_feature_names"][:]
        cat_labels = []

        # Add labels and predictors to cat columns for classification
        if meta_data["task_type"] == "classification":
            # ONLY IF CAT TYPE CHECK HERE
            cat_labels = meta_data["predictors"][:]  # preds of algos should be cat
            cat_labels.append(meta_data["target_name"])  # target is cat
            cat_f.extend(cat_labels)

        to_read_as_cat = {col_name: 'category' for col_name in cat_f}
        meta_dataset = pd.read_csv(file_path_csv, dtype=to_read_as_cat)

        # Post process categories (will be skipped if not classification)
        for col_name in cat_labels:
            meta_dataset[col_name] = meta_dataset[col_name].cat.set_categories(meta_data["class_labels"])

        # -- Init Meta Data
        for md_k in meta_data.keys():
            setattr(self, md_k, meta_data[md_k])

        # Post process special cases
        self.folds = np.array(self.folds)
        self.is_classification = "classification" == meta_data["task_type"]
        self.is_regression = "regression" == meta_data["task_type"]
        self.missing_metadata_in_file = [x for x in self.meta_data_keys if x not in meta_data.keys()]

        # -- Init Datasets
        self.dataset = meta_dataset[self.feature_names + [self.target_name]]
        self.predictions_and_confidences = meta_dataset[self.pred_and_conf_cols]
        self.validation_predictions_and_confidences = meta_dataset[self.validation_predictions_columns +
                                                                   self.validation_confidences_columns]

    def read_folds(self, fold_indicator: np.ndarray):
        """Read a new folds specification. The user must make sure that later data is added according to these folds.

        Parameters
        ----------

        """
        if not isinstance(fold_indicator, np.ndarray):
            raise ValueError("Folds must be passed as numpy array!")

        if fold_indicator.min() != 0:
            raise ValueError("Folds numbers must start from 0. We are all computer scientists here...")

        if fold_indicator.max() != (len(np.unique(fold_indicator)) - 1):
            raise ValueError("Fold values are wrong. Highest fold unequal to unique values - 1")

        if self.n_instances != len(fold_indicator):
            raise ValueError("Fold indicator contains less samples than the dataset.")

        if len(self.predictors) > 0:
            raise ValueError("Chaining Folds even though predictors have been added based on old folds."
                             " Remove the predictor first or pass the folds earlier.")

        self.folds = fold_indicator

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
            meta_data["folds"] = meta_data["folds"].tolist()  # make it serializable

            json.dump(meta_data, f, ensure_ascii=False, indent=4)

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
            val_pred_cols = ["{}{}{}".format(p_name, self.fold_postfix, i) for i in range(10) for p_name in
                             predictor_names]
            val_conf_cols = ["{}{}.{}".format(self.confidence_prefix, n, p_name) for n in
                             self.class_labels for p_name in val_pred_cols]
            self.validation_predictions_and_confidences = self.validation_predictions_and_confidences.drop(
                columns=val_pred_cols + val_conf_cols)

            # If last validation data base model was removed act on it
            if not self.validation_predictions_columns:
                self.use_validation_data = False

    def filter_predictors(self, remove_bad_predictors: bool = True, remove_constant_predictors: bool = False,
                          remove_worse_than_random_predictors: bool = False,
                          score_metric: Optional[Callable] = None, maximize_metric: Optional[bool] = None):
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
        """

        if remove_bad_predictors:
            self.remove_predictors(self.bad_predictors)

        if remove_constant_predictors:
            # Get unique counts
            uc = self.meta_dataset[self.predictors].nunique()
            # Get constant predictors
            constant_predictors = uc[uc == 1].index.tolist()
            self.remove_predictors(constant_predictors)

        if remove_worse_than_random_predictors:
            if score_metric is None:
                raise ValueError("We require a metric to remove worse than random predictors, ",
                                 "so that we can determine predictor performance to be worse than random.")
            if maximize_metric is None:
                raise ValueError("We need to know whether you want to maximize the the metric to remove worse than ",
                                 "random predictors.")
            random_predictions = np.random.choice(self.class_labels, len(self.dataset))
            random_performance = score_metric(self.ground_truth, random_predictions)

            # Get predictor performance (we ignore confidences for this comparison!)
            pred_performance = self.predictions_and_confidences[self.predictors].apply(lambda x: score_metric(
                self.ground_truth, x), axis=0)

            # Get worse than random predictors based on metric (max/min)
            if maximize_metric:
                worse_than_random_predictors = pred_performance[pred_performance < random_performance].index.tolist()
            else:
                worse_than_random_predictors = pred_performance[pred_performance > random_performance].index.tolist()
            self.remove_predictors(worse_than_random_predictors)

    def filter_duplicates_manually(self, min_sim_pre_filter: bool = True, min_sim: float = 0.85):
        """A function that can help you to manually filter duplicates of a metatask.

        Parameters
        ----------
        min_sim_pre_filter : bool, default=True
            Whether you want to pre-filter the possible duplicates based on their similarity using the edit distance.
        min_sim : float, default=0.85
            Minimal percentage of similarity to be considered for manual filtering.
        """
        # -- Delayed Import because it is an optional function (perhaps not used at all)
        # from difflib import SequenceMatcher
        from Levenshtein import ratio as levenshtein_ratio  # use this as it is less radical than sequence matcher

        # -- Get only predictors with high similarity
        similar_predictors = set()
        sim_pred_to_sim_val = {}
        to_remove_predictors = set()
        for outer_pred_name, outer_pred_dsp in self.predictor_descriptions.items():
            # If pred already in the similar list, no need to check if other preds are similar to it.
            #   (If any would be, they would be found in their own loop)
            if outer_pred_name in similar_predictors:
                continue

            # Loop over all predictors and find sim
            for inner_pred_name, inner_pred_dsp in self.predictor_descriptions.items():
                # Skip if the same predictor
                if outer_pred_name == inner_pred_name:
                    continue
                else:
                    # Find difference between the strings
                    len_to_check = len(outer_pred_dsp) if len(outer_pred_dsp) < len(inner_pred_dsp) else len(
                        inner_pred_dsp)
                    for i in range(len_to_check):
                        if outer_pred_dsp[i] != inner_pred_dsp[i]:
                            difference1 = outer_pred_dsp[i:]
                            difference2 = inner_pred_dsp[i:]
                            break
                    else:
                        difference1 = outer_pred_dsp
                        difference2 = inner_pred_dsp

                    # Add both to similar list if higher than min similarity, afterwards skip
                    # sim_val = SequenceMatcher(None, outer_pred_dsp, inner_pred_dsp, autojunk=False).ratio()
                    sim_val = levenshtein_ratio(difference1, difference2)
                    if sim_val >= min_sim or (not min_sim_pre_filter):
                        similar_predictors.add(outer_pred_name)
                        sim_pred_to_sim_val[outer_pred_name] = sim_val
                        similar_predictors.add(inner_pred_name)
                        sim_pred_to_sim_val[inner_pred_name] = sim_val

                        # Validate by hand
                        print("\n\n## Start Similarity Case ##")
                        print("[1]", outer_pred_name, "|", outer_pred_dsp)
                        print("[2]", inner_pred_name, "|", inner_pred_dsp)
                        print("Difference: "
                              "\n     [1] {} ".format(difference1),
                              "\n     [2] {} ".format(difference2))
                        print("Similarity between the difference (edit distance):", sim_val)
                        to_keep = input("Keep 1,2 or both? ('1', '2', 'both' / '\\n') \n")

                        if to_keep == "1":
                            to_remove_predictors.add(inner_pred_name)
                        elif to_keep == "2":
                            to_remove_predictors.add(outer_pred_name)
                        elif to_keep in ["", "both"]:
                            pass
                        else:
                            raise ValueError("Unable to understand your answer!")
                        print("\n## Finished Similarity Case ##")
                        break

        # To list for index usage
        to_remove_predictors = list(to_remove_predictors)

        if to_remove_predictors:
            self.remove_predictors(to_remove_predictors)
            print("Removed: {}".format(to_remove_predictors))
        else:
            print("No predictors removed.")

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

    def split_meta_dataset(self, meta_dataset) -> Tuple[
        pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits the meta dataset into its subcomponents

        Parameters
        ----------
        meta_dataset: self.meta_dataset

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
        """

        return meta_dataset[self.feature_names], meta_dataset[self.target_name], meta_dataset[self.predictors], \
               meta_dataset[self.confidences], meta_dataset[self.validation_predictions_columns], \
               meta_dataset[self.validation_confidences_columns]

    @staticmethod
    def _save_fold_results(y_true, y_pred, fold_idx, out_path, technique_name, classification=True):
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
                                  meta_train_test_split_fraction: float = 0.5, meta_train_test_split_random_state=0,
                                  pre_fit_base_models: bool = False, base_models_with_names: bool = False,
                                  label_encoder=False, fit_technique_on_original_data=False,
                                  preprocessor=None, output_file_path=None, oracle=False,
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
        meta_train_test_split_fraction: float, default=0.5
            The fraction for the meta train/test split.
        meta_train_test_split_random_state: int, default=0
            The randomness for the meta train/test split.
        pre_fit_base_models: bool, default=False
            Whether or not the base models need to be fitted to be passed to the ensemble technique.
        base_models_with_names: bool, default=False
            Whether or not the base models' list should contain the model and its name.
        fit_technique_on_original_data: bool, default=False
            If this is true, the .fit() method of the ensemble is called with X_train and y_train instead
            of X_meta_train and y_meta_train.
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

        # -- Iterate over Folds
        fold_scores = []
        for idx, train_metadata, test_metadata in self.fold_split(return_fold_index=True):

            # -- Get Data from Metatask
            X_train, y_train, _, _, _, _ = self.split_meta_dataset(train_metadata)
            X_test, y_test, test_base_predictions, test_base_confidences, _, _ = self.split_meta_dataset(test_metadata)

            # -- Employ Preprocessing
            if preprocessor is not None:
                X_train = preprocessor.fit_transform(X_train)
                X_test = preprocessor.transform(X_test)

            # -- Split for ensemble technique evaluation
            X_meta_train, X_meta_test, y_meta_train, y_meta_test, = train_test_split(X_test, y_test,
                                                                                     test_size=meta_train_test_split_fraction,
                                                                                     random_state=meta_train_test_split_random_state,
                                                                                     stratify=y_test)

            # -- Build ensemble technique
            base_models = initialize_fake_models(X_train, y_train, X_test, test_base_predictions,
                                                 test_base_confidences, pre_fit_base_models, base_models_with_names,
                                                 label_encoder)

            # -- Probability Calibration
            base_models = probability_calibration_for_faked_models(base_models, X_meta_train, y_meta_train,
                                                                   probability_calibration, pre_fit_base_models)

            ensemble_model = technique(base_models, **technique_args)

            # -- Fit and Predict
            if fit_technique_on_original_data:  # not supported/used currently
                # might not work as intended with fake models
                ensemble_model.fit(X_train, y_train)
            else:
                ensemble_model.fit(X_meta_train, y_meta_train)

            if oracle:
                y_pred_ensemble_model = ensemble_model.oracle_predict(X_meta_test, y_meta_test)
            else:
                y_pred_ensemble_model = ensemble_model.predict(X_meta_test)

            self._save_fold_results(y_meta_test, y_pred_ensemble_model, idx, output_file_path, technique_name)

            # -- Save scores for return
            if return_scores is not None:
                fold_scores.append(return_scores(y_meta_test, y_pred_ensemble_model))

        # -- Return Score
        if return_scores is not None:
            return fold_scores

    # ---- Experimental Functions
    def _exp_get_base_models_for_all_folds(self, pre_fit_base_models: bool = False,
                                           base_models_with_names: bool = False,
                                           label_encoder=False, preprocessor=None):
        """Get Base Models for all Folds | Experimental | For Evaluation and Analysis

        If the metatasks is created using data from openml, this corresponds to something like sklearn's
        corss_val_predict.
        Hence, it represents fake base models as if fitted on the training data of each fold at once/simultaneously.

        To illustrate, given any instance x of X as input, the returned base model will return a prediction for
        the input as if the fake model was only trained on a the training data of the fold for which x is in the
        hold-out set.

        Returns
        -------
        base_models: list of fake base models
        X: original input data
            (preprocessed)
        y: full ground truth
        """
        X, y, base_predictions, base_confidences, _, _ = self.split_meta_dataset(self.meta_dataset)
        if preprocessor is not None:
            X = preprocessor.fit_transform(X)
        base_models = initialize_fake_models(X, y, X, base_predictions, base_confidences, pre_fit_base_models,
                                             base_models_with_names, label_encoder)
        return base_models, X, y

    def _exp_yield_evaluation_data_across_folds(self, meta_train_test_split_fraction,
                                                meta_train_test_split_random_state,
                                                pre_fit_base_models, base_models_with_names, label_encoder,
                                                preprocessor, include_test_data=False):
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

            base_models = initialize_fake_models(X_train, y_train, X_test, test_base_predictions,
                                                 test_base_confidences, pre_fit_base_models, base_models_with_names,
                                                 label_encoder)

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

    def _exp_yield_data_for_base_model_across_folds(self):

        for idx, train_metadata, test_metadata in self.fold_split(return_fold_index=True):
            # -- Get Data from Metatask
            X_train, y_train, _, _, _, _ = self.split_meta_dataset(train_metadata)
            X_test, _, _, _, _, _ = self.split_meta_dataset(test_metadata)

            yield idx, X_train, X_test, y_train
