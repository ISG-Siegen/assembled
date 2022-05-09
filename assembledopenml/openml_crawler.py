import openml
import pandas as pd

from assembledopenml.metaflow import MetaFlow
from assembledopenml.metatask import MetaTask
from typing import List
from assembledopenml.util.data_fetching import chunker


class OpenMLCrawler:
    def __init__(self, openml_metric_name: str = "area_under_roc_curve", maximize_metric: bool = True,
                 min_runs: int = 1, nr_top_flows: int = 50, test_run: bool = False):
        """A Crawler for OpenML that collects all flows (e.g., sklearn pipelines) for a task ID with specific properties

        The flows are filtered based on meta-properties, which Flows must have to be valid and
        crawled from OpenML.

        Parameters
        ----------
        openml_metric_name : {OpenML Metric Names}, str, default="area_under_roc_curve"
            Name of the Metric to judge a Flow's performance .
        maximize_metric : bool, default=True
            Whether or not the metric must be maximized.
        min_runs : int, default=1
            The minimal number of runs a flow must have to be valid (runs in openml can be understood as
            execution of a flow with a different configuration/hyperparameters).
        nr_top_flows: int, default=50
            The number of top-performing flows to select from all found flows {int}.
            If less than nr_top_flows flows are valid, only less than nr_top_flows can be returned.
        test_run: bool, default=False
            If true, the request size is capped to the first 10000 runs to avoid too huge requests.
        """

        self.openml_metric_name = openml_metric_name
        self.maximize_metric = maximize_metric
        self.min_runs = min_runs
        self.nr_top_flows = nr_top_flows
        self.test_run = test_run
        self.test_request_size = 10000 if test_run else None

        # Check if valid metric name
        if openml_metric_name not in openml.evaluations.list_evaluation_measures():
            raise ValueError("Metric Name '{}' does not exit in OpenML!".format(openml_metric_name))

    def _crawl_and_filter_flows(self, openml_task_id: int, evaluation_request_chunks: int = 300) -> List[MetaFlow]:
        """Collect Flows for the task based on the filter setting specified in the initialization."""
        # -- Get all Runs for a Task. This allows us to find all flows via 1 (big) API Call
        task_runs = openml.runs.list_runs(task=[openml_task_id], size=self.test_request_size,
                                          output_format="dataframe")

        # --- Filter Runs
        flows_run_counts = task_runs["flow_id"].value_counts()  # number of runs per flow

        # -- Min Number of Runs Filter
        candidate_flow_ids = flows_run_counts[flows_run_counts >= self.min_runs]

        # -- Re-check Runs for each flow to avoid duplicated runs; Collect best performance of flow (best run)
        flow_to_performance = {}
        flow_name_to_performance = {}
        flow_to_performance_keys_to_del = []
        for flow_id in candidate_flow_ids.index:
            # - Check for duplicates
            runs_for_flow = task_runs[task_runs["flow_id"] == flow_id]
            setups_per_flow = runs_for_flow["setup_id"].value_counts()
            if setups_per_flow.max() > 1:
                # Remove duplicates
                runs_for_flow = runs_for_flow.drop_duplicates("setup_id")
                # Ignore run if is has not enough runs without duplicates
                if self.min_runs > len(runs_for_flow):
                    continue

            # - Check best performance
            run_scores = pd.DataFrame()
            # Collect scores in chunks (because URL too long error otherwise)
            for id_chunk in chunker(runs_for_flow["run_id"].values.tolist(), evaluation_request_chunks):
                chunk_scores = openml.evaluations.list_evaluations(function=self.openml_metric_name, runs=id_chunk,
                                                                   output_format="dataframe",
                                                                   size=None)
                run_scores = pd.concat([run_scores, chunk_scores])
            # Collect best score and its run id
            best_performance_for_flow = run_scores["value"].max() if self.maximize_metric else run_scores["value"].min()
            best_run_id_for_flow = run_scores.iloc[run_scores["value"].argmax(axis=0)]["run_id"]
            # We can use the results from the request above to get the name of the current flow
            flow_name = run_scores.iloc[0]["flow_name"]

            # - Check for duplicated flow versions and guarantee to only take the best of them
            #       remove version from flow name; Assumed format: ".*(X)" whereby X is the version number.
            flow_name_wo_version = flow_name.rsplit("(", 1)[0]
            if flow_name_wo_version not in flow_name_to_performance:
                # Store flow id and performance to compare later
                flow_name_to_performance[flow_name_wo_version] = (flow_id, best_performance_for_flow)
            else:
                f_id, pf = flow_name_to_performance[flow_name_wo_version]
                # Catch duplicate keys to remove later
                if pf < best_performance_for_flow:
                    # We keep the new flow
                    flow_name_to_performance[flow_name_wo_version] = (flow_id, best_performance_for_flow)
                    flow_to_performance_keys_to_del.append(f_id)
                else:
                    # We keep the previous flow
                    flow_to_performance_keys_to_del.append(flow_id)

            # - Save for later
            flow_to_performance[flow_id] = (best_performance_for_flow, best_run_id_for_flow, flow_name)

        # -- Filter version duplicates
        for k in flow_to_performance_keys_to_del:
            del flow_to_performance[k]

        # -- Filter to top-n flows
        top_n_flows = sorted(flow_to_performance.items(), key=lambda x: x[0],
                             reverse=self.maximize_metric)[:self.nr_top_flows]

        # -- Make to list and init metaflow object; and return
        return [MetaFlow(flow_key, values[2], values[0], values[1]) for flow_key, values in
                top_n_flows]

    def _build_flows_for_predictors(self, valid_predictors: List[str]) -> List[MetaFlow]:
        meta_flows = []

        for predictor_name in valid_predictors:
            flow_id, run_id = MetaTask.predictor_name_to_ids(predictor_name)
            meta_flows.append(MetaFlow(flow_id, None, None, run_id))

        return meta_flows

    def run(self, openml_task_id: int) -> MetaTask:
        """Crawl OpenML for valid flows and their predictions.

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

        # -- Crawl Valid Flows
        valid_flows = self._crawl_and_filter_flows(openml_task_id)
        print("Found {} fitting Flows in total.".format(len(valid_flows)))
        meta_task.read_predictions_from_metaflows(valid_flows)

        # -- Fill selection constraints data
        meta_task.read_selection_constraints(self.openml_metric_name, self.maximize_metric, self.min_runs,
                                             self.nr_top_flows, self.test_run)

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
        meta_task.read_selection_constraints(self.openml_metric_name, self.maximize_metric, self.min_runs,
                                             self.nr_top_flows, self.test_run)

        return meta_task
