import openml

from assembledopenml.metaflow import MetaFlow
from assembledopenml.metatask import MetaTask
from typing import List, OrderedDict


class OpenMLCrawler:
    def __init__(self, openml_metric_name: str = "area_under_roc_curve", maximize_metric: bool = True,
                 nr_base_models: int = 50):
        """A Crawler for OpenML that collects top-performing runs (configurations) for a task ID.

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

    def _crawl_and_filter_runs(self, openml_task_id: int):
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

    def _build_flows_for_predictors(self, valid_predictors: List[str]) -> List[MetaFlow]:
        meta_flows = []

        for predictor_name in valid_predictors:
            flow_id, run_id = MetaTask.predictor_name_to_ids(predictor_name)
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

    def _reset_crawler(self):
        self.top_n = []
        self.setup_ids = set()

    def run(self, openml_task_id: int) -> MetaTask:
        """Crawl OpenML for valid runs and their predictions.

        Parameters
        ----------
        openml_task_id : int
            An Task ID from OpenML for which to build a metatask.

        Returns
        -------
        meta_task: MetaTask
            A Metatask created based on the crawler's settings for the OpenML Task provided as input
        """
        # -- Crawl OpenML Task and build metatask
        task = openml.tasks.get_task(openml_task_id)
        meta_task = MetaTask()
        meta_task.read_dataset_from_task(task)

        # -- Crawl Configurations
        self._crawl_and_filter_runs(openml_task_id)
        print("We have found {} base models.".format(len(self.top_n)))
        meta_task.read_predictions_from_metaflows(self.top_n)

        # -- Fill selection constraints data
        meta_task.read_selection_constraints(self.openml_metric_name, self.maximize_metric, self.nr_base_models)

        # -- Rest such that crawler can be re-used
        self._reset_crawler()

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
            A Metatask created based on the crawler's settings for the OpenML Task provided as input
        """

        # -- Crawl OpenML Task and build metatask
        task = openml.tasks.get_task(openml_task_id)
        meta_task = MetaTask()
        meta_task.read_dataset_from_task(task)

        # -- Get predictor data and read into metatask
        valid_flows = self._build_flows_for_predictors(valid_predictors)
        meta_task.read_predictions_from_metaflows(valid_flows)

        # -- Fill selection constraints data
        meta_task.read_selection_constraints(self.openml_metric_name, self.maximize_metric, self.nr_base_models)

        return meta_task
