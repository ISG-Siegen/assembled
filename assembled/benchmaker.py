import os
import glob
import json
from typing import Tuple, Callable, List
from pathlib import Path
import shutil

from assembled.metatask import MetaTask


class BenchMaker:
    """The Benchmark Maker class to build and manage benchmarks for a list of tasks

    Parameters
    ----------
    path_to_metatasks: str
        Path to the directory of metatasks that shall be used to build the benchmark.
    output_path_benchmark_metatask: str
        Path to the directory in which the selected and post-processed metatasks of the benchmark shall be stored.
    tasks_to_use: List[int], default=None
        If not None, the task IDs in the list are used to determine which metatasks to load from path_to_metatasks.
    manual_filter_duplicates: bool, default=False
        If you want to manually filter duplicated base models. If True, an interactive session is started once needed.
    min_number_predictors: int, default=4
        The minimal number of predictors each metatask should have to be included in the benchmark.
    remove_constant_predictors: bool, default=False
        If True, remove constant predictors from the metatask.
    remove_worse_than_random_predictors: bool, default=False
        If True, remove worse than random predictors from the metatask.
    remove_bad_predictors: bool, default=False
        IF True, we remove predictors that were marked to bad during the creation of the metataks.
    metric_info: Tuple[metric:callable, metric_name:str, maximize: bool] scorer metric like, default=None
        The metric information required to determine performance.
        Must include a callable, a name, and whether the metric is to be optimized.
        Must be set if remove_worse_than_random_predictors is True.
    """

    def __init__(self, path_to_metatasks: str, output_path_benchmark_metatask: str, tasks_to_use: List[str] = None,
                 manual_filter_duplicates: bool = False, min_number_predictors: int = 4,
                 remove_constant_predictors: bool = False, remove_worse_than_random_predictors: bool = False,
                 remove_bad_predictors: bool = False, metric_info: Tuple[Callable, str, bool] = None):
        # TODO: add path validation here
        self.valid_task_ids = get_id_and_validate_existing_data(path_to_metatasks)
        if tasks_to_use is not None:
            self.valid_task_ids = [t_id for t_id in self.valid_task_ids if t_id in tasks_to_use]

        self.path_to_metatasks = path_to_metatasks
        self.output_path_benchmark_metatask = output_path_benchmark_metatask
        self.manual_filter_duplicates = manual_filter_duplicates
        self.min_number_predictors = min_number_predictors
        self.remove_constant_predictors = remove_constant_predictors
        self.remove_worse_than_random_predictors = remove_worse_than_random_predictors
        self.remove_bad_predictors = remove_bad_predictors

        if (remove_worse_than_random_predictors is True) and (metric_info is None):
            raise ValueError("Metirc Information must be set if any of "
                             "{} is set to True.".format(["remove_worse_than_random_predictors"]))

        if metric_info is not None:
            self.metric = metric_info[0]
            self.metric_name = metric_info[1]
            self.metric_maximize = metric_info[2]
        else:
            self.metric = None
            self.metric_name = None
            self.metric_maximize = None

        # -- Other
        self.tmp_dir = "bmer_tmp"

    def build_benchmark(self, share_data: str = "no"):
        """ Processes the metatasks according to the BenchMakers initialized settings. Moreover, store the new benchmark
            metatasks in the appropriate repository and create a file (benchmark_details.json) containing all
            relevant details about the benchmark.

        Parameters
        ----------
        share_data: str in {"no", "openml", "share_prediction_data"}, default="no
        Determine the strategy used to share the benchmark data.
        *   "no" - no effort is undertake to make the data sharable. This can be used if the full benchmark task files
                   (.csv and .json) are sharable without any license or distribution issues.
        *   "openml" - We can use the OpenML platform to re-produce metatasks. This assumes that all predictors in a
                       metatask are from OpenML (got via Assembled-OpenML) and that the dataset and task data are from
                       an OpenML task. This allows to share (i.e., re-build) a metatasks by only sharing the
                       benchmark_details.json.
        *   "share_meta_data" - This options can be used when you are not able/allowed to share the dataset but
                                can share the meta data like the prediction data or dataset metadata.
                                One use case would be that you have used a dataset from OpenML, but computed all
                                prediction data locally.

                                    The shared meta data includes validation data (if available).
                                    The shared meta data includes metadata about a dataset (feature names, ...).

                                It will save the data that is to be shared in a .zip file under the
                                output_path_benchmark_metatask directory. This .zip file together with the
                                benchmark_details.json can be used to re-produce metatasks. The dataset is not part of
                                the .zip or benchmark_details.json. To later fill the dataset into a task, our tools can
                                get the dataset (e.g. via an OpenML task ID) or the dataset must be passed to our tools.
        """

        if share_data not in ["no", "openml", "share_prediction_data"]:
            raise ValueError("Reproduce data is not a valid value. Got: {}".format(share_data))

        if share_data == "share_prediction_data":
            path_tmp_dir = Path(self.output_path_benchmark_metatask).joinpath(self.tmp_dir)
            path_tmp_dir.mkdir(exist_ok=True)

        benchmark_valid_tasks = {}
        total_selection_constraints = {}
        selection_constraints_per_task = {}

        # Iterate over tasks
        for task_nr, task_id in enumerate(self.valid_task_ids, start=1):
            mt = MetaTask()
            mt.read_metatask_from_files(self.path_to_metatasks, task_id)
            print("#### Process Task {} for dataset {} ({}/{}) ####".format(mt.openml_task_id, mt.dataset_name,
                                                                            task_nr, len(self.valid_task_ids)))

            # Initial Check for number of predictors
            if len(mt.predictors) < self.min_number_predictors:
                continue

            # Filter using Metatasks functionalities
            mt.filter_predictors(remove_bad_predictors=self.remove_bad_predictors,
                                 remove_constant_predictors=self.remove_constant_predictors,
                                 remove_worse_than_random_predictors=self.remove_worse_than_random_predictors,
                                 score_metric=self.metric, maximize_metric=self.metric_maximize)

            # Filter base models duplicates
            if self.manual_filter_duplicates:
                mt.filter_duplicates_manually(min_sim_pre_filter=True, min_sim=0.85)

            # Post Check for number of predictors
            if len(mt.predictors) < self.min_number_predictors:
                continue

            # -- Save benchmark task
            benchmark_valid_tasks[task_id] = mt.predictors
            mt.to_files(self.output_path_benchmark_metatask)

            # Save selection constrains
            for sel_cons in mt.selection_constraints.keys():
                if sel_cons not in total_selection_constraints:
                    total_selection_constraints[sel_cons] = set()
                total_selection_constraints[sel_cons].add(mt.selection_constraints[sel_cons])
            selection_constraints_per_task[task_id] = mt.selection_constraints

            # -- Make data sharable
            # For "no" and "openml", nothing needs to be done.
            if share_data == "share_prediction_data":
                mt.to_sharable_prediction_data(path_tmp_dir)

        # Post-process selection constraints away from set (otherwise not serializable)
        for sel_cons in total_selection_constraints.keys():
            total_selection_constraints[sel_cons] = list(total_selection_constraints[sel_cons])

        # Inform user about results
        print("Found {} valid benchmarks with the following IDs: {}".format(len(benchmark_valid_tasks),
                                                                            benchmark_valid_tasks.keys()))

        # Post-process sharable data
        if share_data == "share_prediction_data":
            self._post_process_data_sharing(path_tmp_dir)

        # Generate benchmark_detail and save them
        bm_ds = self.generate_benchmark_details(benchmark_valid_tasks, total_selection_constraints,
                                                selection_constraints_per_task)
        file_path_json = os.path.join(self.output_path_benchmark_metatask, "benchmark_details.json")
        with open(file_path_json, 'w', encoding='utf-8') as f:
            json.dump(bm_ds, f, ensure_ascii=False, indent=4)

    @property
    def benchmark_parameters(self):

        bm_paras = ["manual_filter_duplicates", "manual_filter_duplicates", "min_number_predictors",
                    "remove_constant_predictors", "remove_worse_than_random_predictors",
                    "remove_bad_predictors", "metric_name", "metric_maximize"]

        return {k: getattr(self, k) for k in bm_paras}

    def generate_benchmark_details(self, task_ids_to_valid_predictors, total_selection_constraints,
                                   selection_constraints_per_task):

        return {
            "valid_task_ids": list(task_ids_to_valid_predictors.keys()),
            "task_ids_to_valid_predictors": task_ids_to_valid_predictors,
            "total_selection_constraints": total_selection_constraints,
            "selection_constraints_per_task": selection_constraints_per_task,
            "benchmark_search_parameters": self.benchmark_parameters
        }

    def _post_process_data_sharing(self, path_tmp_dir):
        shutil.make_archive(path_tmp_dir, 'zip', path_tmp_dir)
        shutil.rmtree(path_tmp_dir)


def get_id_and_validate_existing_data(path_to_metatasks):
    # -- Get all existing file paths in metatasks directory
    dir_path_csv = os.path.join(path_to_metatasks, "metatask_*.csv")
    dir_path_json = os.path.join(path_to_metatasks, "metatask_*.json")
    data_paths = glob.glob(dir_path_csv)
    meta_data_paths = glob.glob(dir_path_json)

    # -- Merge files for validation
    path_tuples = []
    for csv_path in data_paths:
        csv_name = csv_path[:-4]
        for json_path in meta_data_paths:
            json_name = json_path[:-5]
            if csv_name == json_name:
                path_tuples.append((csv_path, json_path, os.path.basename(csv_name)))
                break

    # - Validate number of pairs
    l_dp = len(data_paths)
    l_mdp = len(meta_data_paths)
    l_pt = len(path_tuples)
    if l_pt < l_mdp or l_pt < l_dp:
        print("Found more files in the preprocessed data directory than file pairs: " +
              "Pairs {}, CSV files: {}, JSON files: {}".format(l_pt, l_dp, l_mdp))

    # - Validate correctness of merge
    for paths_tuple in path_tuples:
        try:
            assert paths_tuple[0][:-4] == paths_tuple[1][:-5]
        except AssertionError:
            raise ValueError("Some data files have wrongly configured names: {} vs. {}".format(paths_tuple[0],
                                                                                               paths_tuple[1]))
    # - Only get task Ids
    return [p_vals[2].rsplit(sep="_", maxsplit=1)[-1] for p_vals in path_tuples]
