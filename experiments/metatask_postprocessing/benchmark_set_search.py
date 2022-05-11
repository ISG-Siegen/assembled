""" Script to build a benchmark with specific additional constrains on the base models from a set of crawled metatasks.

The employed constrains and the default values can be found in the project root's README.
"""

import json
import os
from pysat.examples.hitman import Hitman
from sklearn.model_selection import StratifiedShuffleSplit

from statistics import mean
from itertools import chain
from collections import Counter

from assembledopenml.metatask import MetaTask
from results.data_utils import get_id_and_validate_existing_data
from experiments.metatask_postprocessing.old_baselines_code import VirtualBestAlgorithm, SingleBestAlgorithm
from assembledopenml.compatibility.openml_metrics import OpenMLAUROC


def _vba(test_base_predictions, y_test, test_base_confidences, confs_per_class_per_predictor):
    vba = VirtualBestAlgorithm(test_base_predictions, y_test, stack_method="predict_proba",
                               classification=True, base_est_confidences=test_base_confidences,
                               conf_cols_to_est_label=confs_per_class_per_predictor)
    return vba.predict(), vba.virtual_best_predictors_indices, vba.virtual_best_predictors_indices_sets


def _sba(preprocessed_X_test, y_train, train_base_predictions, test_base_predictions, metric_to_use,
         maximize_metric):
    sba = SingleBestAlgorithm(metric_to_use=metric_to_use, maximize_metric=maximize_metric)
    sba.fit(train_base_predictions, y_train)
    pred = sba.predict(preprocessed_X_test, test_base_predictions)
    return pred, sba.selected_indices


def get_vba_sba_scores(meta_task, met):
    # Get VBA/SBA Scores across all folds for a meta task and a metric
    max_nr_folds = meta_task.max_fold + 1  # +1 because folds start counting from 0

    vba_perf_list = []
    sba_perf_list = []
    sba_selacc_list = []
    predictors_to_vba_share = {pred_name: [] for pred_name in meta_task.predictors}
    full_h_solver = Hitman()

    for idx, _, test_metadata in meta_task.fold_split(return_fold_index=True):
        print("## Fold {}/{} ##".format(idx + 1, max_nr_folds))

        # Get usable part of the predictions and train/test split
        i_X_test, i_y_test, i_test_base_predictions, i_test_base_confidences = mt.split_meta_dataset(test_metadata)
        # TODO, refactor to use normal train_test_split from sklearn in the future
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        train_ids, test_ids = list(sss.split(i_X_test, i_y_test))[0]
        i_y_train = i_y_test.iloc[train_ids]
        i_train_base_predictions = i_test_base_predictions.iloc[train_ids]
        i_X_test = i_X_test.iloc[test_ids]
        i_y_test = i_y_test.iloc[test_ids]
        i_test_base_predictions = i_test_base_predictions.iloc[test_ids]
        i_test_base_confidences = i_test_base_confidences.iloc[test_ids]

        # -- Get Baseline Results
        o_y_pred_vba, _, sel_ind_vba_set = _vba(i_test_base_predictions, i_y_test,
                                                i_test_base_confidences,
                                                meta_task.confs_per_class_per_predictor)
        o_y_pred_sba, sel_ind_sba = _sba(i_X_test, i_y_train, i_train_base_predictions,
                                         i_test_base_predictions, met, score_metric_maximize)

        # -- Gap related values
        vba_perf_list.append(met(i_y_test, o_y_pred_vba))
        sba_perf_list.append(met(i_y_test, o_y_pred_sba))
        n_instances = len(i_y_test)
        sba_selacc_list.append(sum(i in j for i, j in zip(sel_ind_sba, sel_ind_vba_set)) / n_instances)

        # -- Predictor share related values
        flat_list = list(chain(*sel_ind_vba_set.to_list()))
        value_counts = Counter(flat_list)
        # Store the vba-share of ech predictor
        for pred_idx in value_counts.keys():
            pred_name = meta_task.predictors[pred_idx]
            current_share = value_counts[pred_idx] / n_instances
            predictors_to_vba_share[pred_name].append(current_share)

        # -- VBA must have related values (get subset
        unique_combs = set([frozenset(l) for l in sel_ind_vba_set.to_list()])
        for unique_comb in unique_combs:
            full_h_solver.hit(unique_comb)

    # Get final hit set
    vba_must_have_predictors = [meta_task.predictors[min_hit_set_idx] for min_hit_set_idx in full_h_solver.get()]

    # Aggregate share over folds
    for pred_name in predictors_to_vba_share.keys():
        predictors_to_vba_share[pred_name] = mean(predictors_to_vba_share[pred_name])

    return mean(vba_perf_list), mean(sba_perf_list), mean(
        sba_selacc_list), predictors_to_vba_share, vba_must_have_predictors


if __name__ == "__main__":

    # - Subset Search Parameter
    vba_sba_perf_gap_min = 0.05
    vba_sba_sel_gap_min = 0
    only_must_have_predictors = False  # only select the subset of all predictors that are needed to build the VBA
    min_number_predictors = 10
    manual_filter_duplicates = False
    remove_constant_predictors = False
    remove_worse_than_random_predictors = True
    remove_bad_predictors = True
    score_metric = OpenMLAUROC()
    score_metric_maximize = score_metric.maximize
    score_metric_name = score_metric.name

    # - Output
    benchmark_valid_tasks = {}
    selection_constraints = None
    out_base_path = "../../results/benchmark_metatasks"

    # - Input parameter
    base_path = "../../results/metatasks"
    valid_task_ids = get_id_and_validate_existing_data(base_path)

    # Iterate over tasks to gather results
    nr_tasks = len(valid_task_ids)
    for task_nr, task_id in enumerate(valid_task_ids, start=1):
        mt = MetaTask()
        mt.read_metatask_from_files(base_path, task_id)
        print("#### Process Task {} for dataset {} ({}/{}) ####".format(mt.openml_task_id, mt.dataset_name,
                                                                        task_nr, nr_tasks))

        # -- Initial Filter for Predictors
        mt.filter_predictors(remove_bad_predictors=remove_bad_predictors,
                             remove_constant_predictors=remove_constant_predictors,
                             remove_worse_than_random_predictors=remove_worse_than_random_predictors,
                             score_metric=score_metric, maximize_metric=score_metric_maximize)

        # -- Get SBA Score, Get VBA Score, Get VBA Predictor Percentage
        vba_perf, sba_perf, sba_selcacc, pred_vba_share, vba_must_have_preds = get_vba_sba_scores(mt, score_metric)

        # -- After Score Filter
        if (1 - (sba_perf / vba_perf) >= vba_sba_perf_gap_min) and ((1 - sba_selcacc) >= vba_sba_sel_gap_min):

            if manual_filter_duplicates:
                mt.filter_duplicates_manually(min_sim_pre_filter=True, min_sim=0.85)
                # Have to re-run this, as the duplicates-filter step could have changed these values
                _, _, _, _, vba_must_have_preds = get_vba_sba_scores(mt, score_metric)

            # Remove all unnecessary predictors
            if only_must_have_predictors:
                to_remove_predictors = [pred_name for pred_name in mt.predictors
                                        if pred_name not in vba_must_have_preds]
                mt.remove_predictors(to_remove_predictors)

            if len(mt.predictors) >= min_number_predictors:
                # Has enough predictors to be valid
                benchmark_valid_tasks[task_id] = mt.predictors
                mt.to_files(out_base_path)
                selection_constraints = mt.selection_constraints
    # - Output
    print("Benchmark Task Predictors:", benchmark_valid_tasks)
    print("Benchmark Task IDs:", benchmark_valid_tasks.keys())

    # -- Store benchmark metadata
    benchmark_valid_tasks = {
        "valid_task_ids": list(benchmark_valid_tasks.keys()),
        "task_ids_to_valid_predictors": benchmark_valid_tasks,
        "selection_constraints": selection_constraints,
        "benchmark_search_parameters": {
            "vba_sba_perf_gap_min": vba_sba_perf_gap_min,
            "vba_sba_sel_gap_min": vba_sba_sel_gap_min,
            "only_must_have_predictors": only_must_have_predictors,
            "min_number_predictors": min_number_predictors,
            "manual_filter_duplicates": manual_filter_duplicates,
            "remove_constant_predictors": remove_constant_predictors,
            "remove_worse_than_random_predictors": remove_worse_than_random_predictors,
            "remove_bad_predictors": remove_bad_predictors,
            "score_metric": score_metric_name,
            "score_metric_maximize": score_metric_maximize
        }
    }
    file_path_json = os.path.join(out_base_path, "benchmark_details.json")
    with open(file_path_json, 'w', encoding='utf-8') as f:
        json.dump(benchmark_valid_tasks, f, ensure_ascii=False, indent=4)
