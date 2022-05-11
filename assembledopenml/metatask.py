import pandas as pd
import numpy as np
import os
import json

from openml import OpenMLTask
from openml.tasks import TaskType
from assembledopenml.metaflow import MetaFlow
from assembledopenml.compatibility.faked_classifier import FakedClassifier
from typing import List, Tuple, Optional, Callable
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
# from difflib import SequenceMatcher
from Levenshtein import ratio as levenshtein_ratio  # use this as it is less radical than sequence matcher from python
import re


class MetaTask:
    def __init__(self):
        """Metatask, a meta version of a normal OpenML Task

        The Metatask contains the predictions and confidences (e.g. sklearn's predict_proba) of specific OpenML runs
        and the data of the original task. Moreover, additional side information are captured.
        This object is filled via functions and thus it is empty initially
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
        self.folds_indicator = None  # An array where each value represents the fold of an instance (starts from 0)

        # -- for predictions init
        self.predictions_and_confidences = pd.DataFrame()
        self.predictors = []  # List of predictor names
        self.predictor_descriptions = {}  # Predictor names to descriptions
        self.bad_predictors = []  # List of predictor names that have ill-formatted predictions or wrong data
        self.found_confidence_prefixes = set()
        self.predictor_corruptions_details = {}
        self.confidences = []  # List of column names of confidences

        # -- Selection constrains (used to find the data of this metatask)
        self.selection_constraints = {}

        # -- Other
        self.supported_task_types = {TaskType.SUPERVISED_CLASSIFICATION, TaskType.SUPERVISED_REGRESSION}

    @property
    def n_classes(self):
        return len(self.class_labels)

    @property
    def n_instances(self):
        return len(self.dataset)

    @property
    def meta_dataset(self):
        return pd.concat([self.dataset, self.predictions_and_confidences], axis=1)

    @property
    def confidence_prefix(self):
        # Only get the first confidence prefix
        return list(self.found_confidence_prefixes)[0]

    @property
    def meta_data(self):

        meta_data = {
            "openml_task_id": self.openml_task_id,
            "dataset_name": self.dataset_name,
            "target_name": self.target_name,
            "class_labels": self.class_labels,
            "predictors": self.predictors,
            "confidences": self.confidences,
            "predictor_descriptions": self.predictor_descriptions,
            "bad_predictors": self.bad_predictors,
            "confidence_prefix": self.confidence_prefix,  # select only the first
            "feature_names": self.feature_names,
            "cat_feature_names": self.cat_feature_names,
            "folds": self.folds_indicator.tolist(),
            "selection_constraints": self.selection_constraints,
            "task_type": "classification" if self.is_classification else "regression",
            "predictor_corruptions_details": self.predictor_corruptions_details
        }

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
        return int(self.folds_indicator.max())

    @staticmethod
    def predictor_name_to_ids(pred_name):

        if not re.match("^prediction_flow_\d*_run_\d*$", pred_name):
            raise ValueError("Unknown Name format for the predictor: {}".format(pred_name))

        # Split
        _, flow_id, _, run_id = pred_name.rsplit("_", 3)

        return int(flow_id), int(run_id)

    def fold_split(self, return_fold_index=False):
        # Return split copy of metadataset
        for i in range(self.max_fold + 1):
            test_indices = self.folds_indicator == i
            train_indices = self.folds_indicator != i

            if return_fold_index:
                yield i, self.meta_dataset.iloc[train_indices].copy(), self.meta_dataset.iloc[test_indices].copy()
            else:
                yield self.meta_dataset.iloc[train_indices].copy(), self.meta_dataset.iloc[test_indices].copy()

    def split_meta_dataset(self, meta_dataset) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
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
               meta_dataset[self.confidences]

    def _init_dataset_information(self, dataset: pd.DataFrame, target_name: str, class_labels: List[str],
                                  feature_names: List[str], cat_feature_names: List[str], task_type: int,
                                  openml_task_id: TaskType, folds_indicator: np.ndarray, dataset_name: str):
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
        task_type: TaskType
            OpenML TaskType for the task.
        openml_task_id: int
            OpenML Task ID
        folds_indicator: np.ndarray
            Array of length (n_samples,) indicating the folds for each instances (starting from 0)
        dataset_name: str
            Name of the dataset
        """
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.target_name = target_name
        self.class_labels = sorted(class_labels)
        self.feature_names = feature_names
        self.cat_feature_names = cat_feature_names
        self.openml_task_id = openml_task_id
        self.folds_indicator = folds_indicator

        # Handle task type
        self.task_type = task_type
        if task_type not in self.supported_task_types:
            raise ValueError("Unsupported OpenML Task Type ID {} was found.".format(task_type))
        self.is_classification = task_type == TaskType.SUPERVISED_CLASSIFICATION
        self.is_regression = task_type == TaskType.SUPERVISED_REGRESSION

        self._check_and_init_ground_truth()

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

    def read_dataset_from_task(self, openml_task: OpenMLTask):
        """ Fill the metatask's dataset using an initialized OpenMLTask

        Parameters
        ----------
        openml_task : OpenMLTask
            The OpenML Task object for which we shall build a metatask.
        """
        # -- Get relevant data from task
        openml_dataset = openml_task.get_dataset()
        dataset_name = openml_dataset.name
        dataset, _, cat_indicator, feature_names = openml_dataset.get_data()
        target_name = openml_task.target_name
        class_labels = openml_task.class_labels
        # - Get Cat feature names
        cat_feature_names = [f_name for cat_i, f_name in zip(cat_indicator, feature_names) if
                             (cat_i == 1) and (f_name != target_name)]
        feature_names.remove(target_name)  # Remove only afterwards, as indicator includes class
        task_type = openml_task.task_type_id
        task_id = openml_task.task_id

        # - Handle Folds
        openml_task.split = openml_task.download_split()
        folds_indicator = np.empty(len(dataset))
        for i in range(openml_task.split.folds):
            # FIXME Ignores repetitions for now (only take first repetitions if multiple are available)
            _, test = openml_task.get_train_test_split_indices(fold=i)
            folds_indicator[test] = int(i)

        # -- Fill object with values
        self._init_dataset_information(dataset, target_name, class_labels, feature_names, cat_feature_names,
                                       task_type, task_id, folds_indicator, dataset_name)

    def read_predictions_from_metaflows(self, metaflows: List[MetaFlow]):
        """Fill prediction data from a list of metaflow objects

        Parameters
        ----------
        metaflows : List[MetaFlow]
            A list of metaflow object of which the prediction data shall be stored in the metatask
        """

        # For each flow, parse prediction and store it
        for meta_flow in metaflows:
            print("#### Process Flow {} + Run {} ####".format(meta_flow.flow_id, meta_flow.run_id))

            # -- Setup Column Names for MetaDataset
            col_name = "prediction_flow_{}_run_{}".format(meta_flow.flow_id, meta_flow.run_id)
            conf_col_names = ["confidence.{}.{}".format(n, col_name) for n in self.class_labels]
            meta_flow.name = col_name

            # -- Get predictions
            meta_flow.get_predictions_data(self.class_labels)

            # - Check if file_y_ture is corrupted (and thus potentially the predictions)
            if sum(meta_flow.file_ground_truth != self.ground_truth) > 0:
                # ground truth of original data differs to predictions file y_ture
                # -> Store it and ignore it
                meta_flow.file_ground_truth_corrupted = True

            # -- Merge (this way because performance warning otherwise)
            re_names = {meta_flow.confidences.columns[i]: n for i, n in enumerate(conf_col_names)}
            self.predictions_and_confidences = pd.concat([self.predictions_and_confidences,
                                                          meta_flow.predictions.rename(col_name),
                                                          meta_flow.confidences.rename(re_names, axis=1)], axis=1)

            # Add to relevant storage
            self.predictors.append(col_name)
            self.confidences.extend(conf_col_names)
            self.predictor_descriptions[col_name] = meta_flow.description
            self.found_confidence_prefixes.add(meta_flow.conf_prefix)

        # Collect meta data about predictors
        self.bad_predictors.extend([mf.name for mf in metaflows if mf.is_bad_flow])
        self.predictor_corruptions_details.update({mf.name: mf.corruption_details for mf in metaflows
                                                   if (mf.file_ground_truth_corrupted or mf.confidences_corrupted)})

    def read_selection_constraints(self, openml_metric_name: str, maximize_metric: bool, nr_base_models: int):
        """Fill the constrains used to build the metatask

        Parameters
        ----------
        openml_metric_name : {OpenML Metric Names}, str, default="area_under_roc_curve"
            Name of the Metric to judge a Flow's performance .
        maximize_metric : bool, default=True
            Whether or not the metric must be maximized.
        nr_base_models: int, default=50
            The number of top-performing configurations (runs).
            If less than nr_base_models runs exist, only less than nr_base_models can be returned.
        """
        self.selection_constraints["openml_metric_name"] = openml_metric_name
        self.selection_constraints["maximize_metric"] = maximize_metric
        self.selection_constraints["nr_base_models"] = nr_base_models

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
        self.openml_task_id = openml_task_id
        self.dataset_name = meta_data["dataset_name"]
        self.target_name = meta_data["target_name"]
        self.class_labels = meta_data["class_labels"]
        self.predictors = meta_data["predictors"]
        self.confidences = meta_data["confidences"]
        self.predictor_descriptions = meta_data["predictor_descriptions"]
        self.bad_predictors = meta_data["bad_predictors"]
        self.found_confidence_prefixes = {meta_data["confidence_prefix"]}
        self.feature_names = meta_data["feature_names"]
        self.cat_feature_names = meta_data["cat_feature_names"]
        self.folds_indicator = np.array(meta_data["folds"])
        self.selection_constraints = meta_data["selection_constraints"]
        self.is_classification = "classification" == meta_data["task_type"]
        self.is_regression = "regression" == meta_data["task_type"]
        self.predictor_corruptions_details = meta_data["predictor_corruptions_details"]
        self.task_type = TaskType.SUPERVISED_CLASSIFICATION if self.is_classification else TaskType.SUPERVISED_REGRESSION

        # -- Init Datasets
        self.dataset = meta_dataset[self.feature_names + [self.target_name]]
        self.predictions_and_confidences = meta_dataset[self.pred_and_conf_cols]

    def read_randomness(self, random_state):
        self.test_split_random_state = random_state

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
            json.dump(self.meta_data, f, ensure_ascii=False, indent=4)

    # -- Post Processing
    def remove_predictors(self, predictor_names):
        rel_conf_and_pred_cols = self.get_pred_and_conf_cols(predictor_names)
        only_conf_cols = [ele for ele in rel_conf_and_pred_cols if ele not in predictor_names]

        # Remove predictors from all parts of the metatask object
        self.predictions_and_confidences = self.predictions_and_confidences.drop(columns=rel_conf_and_pred_cols)
        self.predictors = [pred for pred in self.predictors if pred not in predictor_names]
        self.confidences = [ele for ele in self.confidences if ele not in only_conf_cols]
        for pred_name in predictor_names:
            # save delete, does not raise key error if not in dict if we use pop here
            self.predictor_descriptions.pop(pred_name, None)
            self.predictor_corruptions_details.pop(pred_name, None)

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
        score_metric : openml metric function, default=None
            The metric function used to determine if a predictors is worse than a random predictor.
            Special format required due to OpenML's metrics.
        maximize_metric : bool, default=None
            Whether the metric computed by the metric function passed by score_metric is to be maximized or not.
        """

        if remove_bad_predictors:
            self.remove_predictors(self.bad_predictors)
            self.bad_predictors = []

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
    @staticmethod
    def _build_fake_models(X_train, y_train, X_test, known_predictions, known_confidences, pre_fit_base_models,
                           base_models_with_names, label_encoder):
        # Expect the predictions/confidences on the whole meta-data as input. Whereby meta-data is the data passed to
        #   the ensemble method.

        faked_base_models = []

        for model_name in list(known_predictions):  # known predictions is a dataframe
            model_confidences = known_confidences[["confidence.{}.{}".format(class_name, model_name)
                                                   for class_name in np.unique(y_train)]]
            model_predictions = known_predictions[model_name]
            fc = FakedClassifier(X_test, model_predictions, model_confidences, label_encoder=label_encoder)

            # -- Set fitted or not (sklearn vs. deslib)
            if pre_fit_base_models:
                if isinstance(X_train, pd.DataFrame) and isinstance(y_train, pd.Series):
                    fc.fit(X_train.to_numpy(), y_train.to_numpy())
                elif isinstance(y_train, pd.Series):
                    fc.fit(X_train, y_train.to_numpy())
                else:
                    raise ValueError("Unsupported Types for X_train or y_train: " +
                                     "X_train type is {}; y_train type is {}".format(type(X_train), type(y_train)))

            # -- Set result output (sklearn vs. deslib)
            res = fc
            if base_models_with_names:
                res = (model_name, res)

            faked_base_models.append(res)

        return faked_base_models

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

    def _probability_calibration(self, base_models, X_meta_train, y_meta_train, probability_calibration,
                                 pre_fit_base_models):

        # -- Simply return base models without changes
        if probability_calibration == "no":
            return base_models

        # -- Select calibration method
        if probability_calibration != "auto":
            # We assume the input has been validated and only "sigmoid", "isotonic" are possible options
            cal_method = probability_calibration
        else:
            # TODO-FUTURE: perhaps add something that selects method based on base model types?

            # Select method based on number of instances
            cal_method = "isotonic" if self.n_instances > 1100 else "sigmoid"

        # --Build calibrated base models
        cal_base_models = []
        for bm_data in base_models:

            # - Select the base model
            if isinstance(bm_data, tuple):
                bm = bm_data[1]
            else:
                bm = bm_data

            # - Determine how to process the base models
            if pre_fit_base_models:
                cal_bm = CalibratedClassifierCV(bm, method=cal_method, cv="prefit").fit(X_meta_train, y_meta_train)
            else:
                # With cv=2 we have less overhead with fake base models.
                # Once our current fake base model structure changes, we need to change this as well.
                cal_bm = CalibratedClassifierCV(bm, method=cal_method, ensemble="False", cv=2)

            # - Set base model
            if isinstance(bm_data, tuple):
                cal_base_models.append((bm_data[0], cal_bm))
            else:
                cal_base_models.append(cal_bm)

        return cal_base_models

    def run_ensemble_on_all_folds(self, technique, technique_args: dict, technique_name,
                                  meta_train_test_split_fraction: float = 0.5, meta_train_test_split_random_state=0,
                                  pre_fit_base_models: bool = False, base_models_with_names: bool = False,
                                  label_encoder=False, fit_technique_on_original_data=False,
                                  preprocessor=None, output_file_path=None, oracle=False,
                                  probability_calibration="no"):
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
            of X_meta_train and y_meta_train. [NOT USED CURRENTLY DUE TO FAKED BASE MODELS]
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
        """
        # TODO -- Parameter Preprocessing / Checking
        #   Add safety check for file path here or something
        #   Check if probability_calibration has correct string names

        # -- Iterate over Folds
        for idx, train_metadata, test_metadata in self.fold_split(return_fold_index=True):

            # -- Get Data from Metatask
            X_train, y_train, _, _ = self.split_meta_dataset(train_metadata)
            X_test, y_test, test_base_predictions, test_base_confidences = self.split_meta_dataset(test_metadata)

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
            base_models = self._build_fake_models(X_train, y_train, X_test, test_base_predictions,
                                                  test_base_confidences, pre_fit_base_models, base_models_with_names,
                                                  label_encoder)

            # -- Probability Calibration
            base_models = self._probability_calibration(base_models, X_meta_train, y_meta_train,
                                                        probability_calibration, pre_fit_base_models)

            ensemble_model = technique(base_models, **technique_args)

            # -- Fit and Predict
            # if fit_technique_on_original_data:  # not supported currently due to calibration and fake models usage
            #     ensemble_model.fit(X_train, y_train)
            # else:
            ensemble_model.fit(X_meta_train, y_meta_train)

            if oracle:
                y_pred_ensemble_model = ensemble_model.oracle_predict(X_meta_test, y_meta_test)
            else:
                y_pred_ensemble_model = ensemble_model.predict(X_meta_test)

            self._save_fold_results(y_meta_test, y_pred_ensemble_model, idx, output_file_path, technique_name)

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
        X, y, base_predictions, base_confidences = self.split_meta_dataset(self.meta_dataset)
        if preprocessor is not None:
            X = preprocessor.fit_transform(X)
        base_models = self._build_fake_models(X, y, X, base_predictions, base_confidences, pre_fit_base_models,
                                              base_models_with_names, label_encoder)
        return base_models, X, y

    def _exp_yield_base_models_across_folds(self, meta_train_test_split_fraction, meta_train_test_split_random_state,
                                            pre_fit_base_models, base_models_with_names, label_encoder, preprocessor):
        for idx, train_metadata, test_metadata in self.fold_split(return_fold_index=True):
            X_train, y_train, _, _ = self.split_meta_dataset(train_metadata)
            X_test, y_test, test_base_predictions, test_base_confidences = self.split_meta_dataset(test_metadata)

            if preprocessor is not None:
                X_train = preprocessor.fit_transform(X_train)
                X_test = preprocessor.transform(X_test)

            X_meta_train, X_meta_test, y_meta_train, y_meta_test = train_test_split(X_test, y_test,
                                                                                    test_size=meta_train_test_split_fraction,
                                                                                    random_state=meta_train_test_split_random_state,
                                                                                    stratify=y_test)

            base_models = self._build_fake_models(X_train, y_train, X_test, test_base_predictions,
                                                  test_base_confidences, pre_fit_base_models, base_models_with_names,
                                                  label_encoder)

            yield base_models, X_meta_train, X_meta_test, y_meta_train, y_meta_test
