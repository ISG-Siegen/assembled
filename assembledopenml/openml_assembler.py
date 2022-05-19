import openml
import numpy as np
from openml.tasks import OpenMLTask

from assembledopenml.metaflow import MetaFlow
from assembled.metatask import MetaTask
from typing import List, OrderedDict
import re


class OpenMLAssembler:
    def __init__(self, openml_metric_name: str = "area_under_roc_curve", maximize_metric: bool = True,
                 nr_base_models: int = 50):
        """A wrapper for OpenML that collects top-performing runs (configurations) for a task ID.

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

        self.openml_metric_name = openml_metric_name
        self.maximize_metric = maximize_metric
        self.nr_base_models = nr_base_models

        # Check if valid metric name
        if openml_metric_name not in openml.evaluations.list_evaluation_measures():
            raise ValueError("Metric Name '{}' does not exit in OpenML!".format(openml_metric_name))

        # -- For later
        self.setup_ids = set()
        self.top_n = []

    def _select_top_unique_configurations(self,
                                          openml_evaluations_dict: OrderedDict[
                                              int, openml.evaluations.OpenMLEvaluation]):
        """Select the top performing non-duplicated configurations"""

        for openml_evaluation in openml_evaluations_dict.values():

            if len(self.top_n) == self.nr_base_models:
                # We stop this loop if we have found enough base models
                break

            # Get relevant data
            setup_id = openml_evaluation.setup_id
            eval_perf = openml_evaluation.value

            # Skip if we found this configuration already
            if setup_id in self.setup_ids:
                # Assumption 1: setup ids are unique across a task for OpenML. This seems to be the case.
                # Assumption 2: the performance of two runs with identical setup should be equal
                #       !This is not guaranteed due to randomness and since we do not know the random seed!
                #       However, we ignore it. The validation scheme should account for randomness.
                #           (AutoML would not run configurations multiple times)
                continue

            self.setup_ids.add(setup_id)
            self.top_n.append(MetaFlow(openml_evaluation.flow_id, openml_evaluation.flow_name, eval_perf,
                                       openml_evaluation.run_id))

    def _collect_and_filter_runs(self, openml_task_id: int):
        """Collect Runs for the task based on the filter setting specified in the initialization."""
        sort_order = "desc" if self.maximize_metric else "asc"
        offset = 0
        request_size = 1000
        # Batch over the flows (if test run break after first batch)
        while True:
            # Get batched runs/evaluations for this task (not unique).
            #   We query twice the amount of evaluations as needed to make sure that enough unique
            #       setup ids are found using 1 request.
            batched_evaluations = openml.evaluations.list_evaluations(function=self.openml_metric_name,
                                                                      tasks=[openml_task_id], sort_order=sort_order,
                                                                      size=request_size, offset=offset)
            offset += request_size

            # Filter duplicated configurations
            self._select_top_unique_configurations(batched_evaluations)

            # Immediate Breaks
            if (len(self.top_n) == self.nr_base_models) or (len(batched_evaluations) != request_size):
                # Break if we have found enough models or if we have crawled all data on openml
                del batched_evaluations
                break

            print("Found only {} unique runs in the top {} runs.".format(len(self.top_n), offset))
        self._validate_top_n()
        print(len(self.top_n))

    @staticmethod
    def predictor_name_to_ids(pred_name):

        if not re.match(r"^prediction_flow_\d*_run_\d*$", pred_name):
            raise ValueError("Unknown Name format for the predictor: {}".format(pred_name))

        # Split
        _, flow_id, _, run_id = pred_name.rsplit("_", 3)

        return int(flow_id), int(run_id)

    def _build_flows_for_predictors(self, valid_predictors: List[str]) -> List[MetaFlow]:
        meta_flows = []

        for predictor_name in valid_predictors:
            flow_id, run_id = self.predictor_name_to_ids(predictor_name)
            meta_flows.append(MetaFlow(flow_id, None, None, run_id))

        return meta_flows

    def _validate_top_n(self):
        # Quick checker to validate top-n has intended content

        if len(self.top_n) > self.nr_base_models:
            raise ValueError("Too many base models have been added to top_n somehow.")

        # Check uniqueness of run + flow id
        combinations_seen = set()
        for mf in self.top_n:
            key = "{},{}".format(mf.flow_id, mf.run_id)
            if key in combinations_seen:
                raise ValueError("Non-unique run-flow combination: {}".format(key))
            combinations_seen.add(key)

    def _reset(self):
        self.top_n = []
        self.setup_ids = set()

    def run(self, openml_task_id: int) -> MetaTask:
        """Search through OpenML for valid runs and their predictions.

        Parameters
        ----------
        openml_task_id : int
            An Task ID from OpenML for which to build a metatask.

        Returns
        -------
        meta_task: MetaTask
            A Metatask created based on the collector's settings for the OpenML Task provided as input
        """
        # -- Get OpenML Task and build metatask
        task = openml.tasks.get_task(openml_task_id)
        meta_task = MetaTask()
        meta_task = init_dataset_from_task(meta_task, task)

        # -- Collect Configurations
        self._collect_and_filter_runs(openml_task_id)
        print("We have found {} base models.".format(len(self.top_n)))
        meta_task = init_base_models_from_metaflows(meta_task, self.top_n)

        # -- Fill selection constraints data
        meta_task.read_selection_constraints({"openml_metric_name": self.openml_metric_name,
                                              "maximize_metric": self.maximize_metric,
                                              "nr_base_models": self.nr_base_models})
        meta_task.read_randomness("OpenML")
        # -- Rest such that .run() can be re-used
        self._reset()

        return meta_task

    def rebuild(self, openml_task_id: int, valid_predictors: List[str]) -> MetaTask:
        """Rebuild a metatask from a list of predictors

        Parameters
        ----------
        openml_task_id : int
            An Task ID from OpenML for which to build a metatask.
        valid_predictors: List[str]
            A list of predictor names (in the metatask format) which shall be part of the metatask

        Returns
        -------
        meta_task: MetaTask
            A Metatask created based on the collector's settings for the OpenML Task provided as input
        """

        # -- Get OpenML Task and build metatask
        task = openml.tasks.get_task(openml_task_id)
        meta_task = MetaTask()
        meta_task = init_dataset_from_task(meta_task, task)

        # -- Get predictor data and read into metatask
        valid_flows = self._build_flows_for_predictors(valid_predictors)
        meta_task = init_base_models_from_metaflows(meta_task, valid_flows)

        # -- Fill selection constraints data
        meta_task.read_selection_constraints({"openml_metric_name": self.openml_metric_name,
                                              "maximize_metric": self.maximize_metric,
                                              "nr_base_models": self.nr_base_models})
        meta_task.read_randomness("OpenML")

        return meta_task


def init_dataset_from_task(meta_task: MetaTask, openml_task):
    """ Fill the metatask's dataset using an initialized OpenMLTask

    Parameters
    ----------
    openml_task : OpenMLTask or int
        The OpenML Task object for which we shall build a metatask.
        If int, we will first get the OpenMLTask for that id
    """
    if isinstance(openml_task, int):
        openml_task = openml.tasks.get_task(openml_task)

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
    task_id = openml_task.task_id

    if openml_task.task_type_id is openml.tasks.TaskType.SUPERVISED_CLASSIFICATION:
        task_type = "classification"
    elif openml_task.task_type_id == openml.tasks.TaskType.SUPERVISED_REGRESSION:
        task_type = "regression"
    else:
        raise ValueError("Unknown or not supported openml task type id: {}".format(openml_task.task_type_id))

    # - Handle Folds
    openml_task.split = openml_task.download_split()
    folds_indicator = np.empty(len(dataset))
    for i in range(openml_task.split.folds):
        # FIXME Ignores repetitions for now (only take first repetitions if multiple are available)
        _, test = openml_task.get_train_test_split_indices(fold=i)
        folds_indicator[test] = int(i)

    # -- Fill object with values
    meta_task.init_dataset_information(dataset, target_name, class_labels, feature_names, cat_feature_names,
                                       task_type, task_id, folds_indicator, dataset_name)

    return meta_task


def init_base_models_from_metaflows(meta_task: MetaTask, metaflows: List[MetaFlow]):
    """Fill prediction data from a list of metaflow objects into a metatask object

    Parameters
    ----------
    metaflows : List[MetaFlow]
        A list of metaflow object of which the prediction data shall be stored in the metatask
    """

    # For each flow, parse prediction and store it
    found_confidence_prefixes = set()
    for meta_flow in metaflows:
        print("#### Process Flow {} + Run {} ####".format(meta_flow.flow_id, meta_flow.run_id))

        # -- Setup Column Names for MetaDataset
        name_to_use = "prediction_flow_{}_run_{}".format(meta_flow.flow_id, meta_flow.run_id)

        # -- Get prediction data
        meta_flow.get_predictions_data(meta_task.class_labels)

        # - Check if file_y_ture is corrupted (and thus potentially the predictions)
        if sum(meta_flow.file_ground_truth != meta_task.ground_truth) > 0:
            # ground truth of original data differs to predictions file y_ture
            # -> Store it and ignore it
            meta_flow.file_ground_truth_corrupted = True

        if meta_flow.file_ground_truth_corrupted or meta_flow.confidences_corrupted:
            corruptions_details = meta_flow.corruption_details
        else:
            corruptions_details = None

        meta_task.add_predictor(name_to_use, meta_flow.predictions, meta_flow.confidences,
                                conf_class_labels=meta_task.class_labels,
                                predictor_description=meta_flow.description, bad_predictor=meta_flow.is_bad_flow,
                                corruptions_details=corruptions_details)

        # Other
        found_confidence_prefixes.add(meta_flow.conf_prefix)

    # TODO, decide what to do with this: print(found_confidence_prefixes)

    return meta_task
