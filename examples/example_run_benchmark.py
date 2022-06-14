""" Example Benchmark with Simulated Ensemble Techniques and Metatasks created by Assembled-OpenML

The following examples show how to use a metatask of Assembled-OpenML to evaluate different ensemble techniques on a
benchmark. The predictions of the techniques are stored and can be evaluated using the scripts in the evaluation
directory.
"""

import pandas as pd
import numpy as np
from numpy.random import RandomState
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from assembledopenml.compatability.openml_metrics import openml_area_under_roc_curve
from assembledopenml.metatask import MetaTask
from experiments.data_utils import get_preprocessing_function, get_valid_benchmark_ids
from example_algorithms.ensembles import SimulatedStackingClassifier, SimulatedDynamicClassifierSelector, \
    SimulatedVotingClassifier, SimulatedDynamicEnsembleSelector, SimulatedEnsembleSelector
from example_algorithms.baselines import VirtualBestAlgorithm, SingleBestAlgorithm

# Random State for Reproducibility
rng = RandomState(571355410)


def stacking_classifier(preprocessed_X_train, preprocessed_X_test, y_train, train_base_predictions,
                        test_base_predictions, train_base_confidences, test_base_confidences,
                        confs_per_class_per_predictor):
    # Fit and Predict
    est = SimulatedStackingClassifier(LogisticRegression(random_state=rng, max_iter=1000), passthrough=False,
                                      stack_method="predict_proba",  # "predict"
                                      conf_cols_to_est_label=confs_per_class_per_predictor)
    est.fit(preprocessed_X_train, y_train, train_base_predictions, base_est_confidences=train_base_confidences)
    return est.predict(preprocessed_X_test, test_base_predictions,
                       base_est_confidences=test_base_confidences)


def voting_classifier(preprocessed_X_test, test_base_predictions, test_base_confidences, confs_per_class_per_predictor):
    # Fit and Predict
    est = SimulatedVotingClassifier(voting="hard",  # "soft"
                                    conf_cols_to_est_label=confs_per_class_per_predictor)
    return est.predict(preprocessed_X_test, test_base_predictions,
                       base_est_confidences=test_base_confidences)


def dcs(preprocessed_X_train, preprocessed_X_test, y_train, train_base_predictions,
        test_base_predictions, train_base_confidences, confs_per_class_per_predictor):
    est = SimulatedDynamicClassifierSelector(RandomForestRegressor(random_state=rng),
                                             conf_cols_to_est_label=confs_per_class_per_predictor,
                                             stack_method="predict_proba"  # "predict"
                                             )
    est.fit(preprocessed_X_train, y_train, train_base_predictions, base_est_confidences=train_base_confidences)
    pred = est.predict(preprocessed_X_test, test_base_predictions)
    return pred, est.selected_indices


def dcs_vba(test_base_predictions, y_test, test_base_confidences, confs_per_class_per_predictor):
    vba = VirtualBestAlgorithm(test_base_predictions, y_test, stack_method="predict_proba",
                               classification=True, base_est_confidences=test_base_confidences,
                               conf_cols_to_est_label=confs_per_class_per_predictor)
    return vba.predict(), vba.virtual_best_predictors_indices, vba.virtual_best_predictors_indices_sets


def dcs_sba(preprocessed_X_test, y_train, train_base_predictions, test_base_predictions, metric_to_use,
            maximize_metric):
    sba = SingleBestAlgorithm(metric_to_use=metric_to_use, maximize_metric=maximize_metric)
    sba.fit(train_base_predictions, y_train)
    pred = sba.predict(preprocessed_X_test, test_base_predictions)
    return pred, sba.selected_indices


def des(preprocessed_X_train, preprocessed_X_test, y_train, train_base_predictions,
        test_base_predictions, train_base_confidences, confs_per_class_per_predictor):
    est = SimulatedDynamicEnsembleSelector(RandomForestRegressor(random_state=rng),
                                           conf_cols_to_est_label=confs_per_class_per_predictor,
                                           stack_method="predict_proba"  # "predict"
                                           )
    est.fit(preprocessed_X_train, y_train, train_base_predictions, base_est_confidences=train_base_confidences)
    pred = est.predict(preprocessed_X_test, test_base_predictions)
    return pred


def ensemble_selection(y_train, train_base_confidences, test_base_confidences, metric_to_use,
                       maximize_metric, confs_per_class_per_predictor):
    es = SimulatedEnsembleSelector(50, confs_per_class_per_predictor, metric_to_use=metric_to_use,
                                   maximize_metric=maximize_metric, random_state=rng)
    es.fit(train_base_confidences, y_train)
    return es.predict(test_base_confidences)


def vbe_wrapper(preprocessed_X_test, y_test, test_base_predictions, test_base_confidences,
                confs_per_class_per_predictor):
    """Virtual Best Ensemble

    Non-real oracle-like predictor to represent the case where a weighted ensemble method (like stacking) found the
    optimal set of weights for the test data on the training data.

    For simplicity, we assume that learning the weights on the test data finds an optimal set of weights.
    """
    est = SimulatedStackingClassifier(LogisticRegression(random_state=rng, max_iter=1000), passthrough=False,
                                      stack_method="predict_proba",  # "predict"
                                      conf_cols_to_est_label=confs_per_class_per_predictor)
    est.fit(preprocessed_X_test, y_test, test_base_predictions, base_est_confidences=test_base_confidences)
    return est.predict(preprocessed_X_test, test_base_predictions,
                       base_est_confidences=test_base_confidences)


def get_fold_results(result_values):
    return np.array([res_tup[1] for res_tup in result_values]).T


if __name__ == "__main__":

    # Input parameter
    valid_task_ids = get_valid_benchmark_ids()  # alternatively set ids by hand, e.g.: [3913, 3560]
    score_metric = openml_area_under_roc_curve
    score_metric_maximize = True
    fold_split_test_split_frac = 0.5

    # Iterate over tasks to gather results
    nr_tasks = len(valid_task_ids)
    for task_nr, task_id in enumerate(valid_task_ids, start=1):
        mt = MetaTask()
        mt.read_metatask_from_files("../results/benchmark_metatasks", task_id)
        print("#### Process Task {} for dataset {} ({}/{}) ####".format(mt.openml_task_id, mt.dataset_name,
                                                                        task_nr, nr_tasks))
        out_path = "../results/benchmark_output/results_for_metatask_{}.csv".format(task_id)
        # -- Get Preprocessing Function
        pre_proc = get_preprocessing_function(mt.cat_feature_names, mt.non_cat_feature_names)

        # -- Iterate over Folds
        res_cols = None
        max_nr_folds = mt.max_fold + 1  # +1 because folds start counting from 0
        for idx, _, test_metadata in mt.fold_split(return_fold_index=True):
            print("## Fold {}/{} ##".format(idx + 1, max_nr_folds))

            # Results
            res_values = []

            # Get usable part of the predictions
            i_X_test, i_y_test, i_test_base_predictions, i_test_base_confidences = mt.split_meta_dataset(test_metadata)

            # Split
            i_X_train, i_X_test, i_y_train, i_y_test, i_train_base_predictions, i_test_base_predictions, \
            i_train_base_confidences, i_test_base_confidences = train_test_split(i_X_test, i_y_test,
                                                                                 i_test_base_predictions,
                                                                                 i_test_base_confidences,
                                                                                 test_size=fold_split_test_split_frac,
                                                                                 random_state=rng,
                                                                                 stratify=i_y_test)

            # Apply preprocessing
            i_preprocessed_X_train, i_preprocessed_X_test = pre_proc(i_X_train, i_X_test)

            # -- Get Predictions from different methods
            o_y_pred_stack_classifier = stacking_classifier(i_preprocessed_X_train, i_preprocessed_X_test, i_y_train,
                                                            i_train_base_predictions, i_test_base_predictions,
                                                            i_train_base_confidences, i_test_base_confidences,
                                                            mt.confs_per_class_per_predictor)
            res_values.append(("StackingClassifier", o_y_pred_stack_classifier))

            o_y_pred_voting_classifier = voting_classifier(i_preprocessed_X_test, i_test_base_predictions,
                                                           i_test_base_confidences,
                                                           mt.confs_per_class_per_predictor)
            res_values.append(("VotingClassifier", o_y_pred_voting_classifier))

            o_y_pred_ensemble_selection = ensemble_selection(i_y_train, i_train_base_confidences,
                                                             i_test_base_confidences, score_metric,
                                                             score_metric_maximize,
                                                             mt.confs_per_class_per_predictor)
            res_values.append(("EnsembleSelection", o_y_pred_ensemble_selection))

            o_y_pred_dcs, _ = dcs(i_preprocessed_X_train,
                                  i_preprocessed_X_test, i_y_train,
                                  i_train_base_predictions,
                                  i_test_base_predictions,
                                  i_train_base_confidences,
                                  mt.confs_per_class_per_predictor)
            res_values.append(("DCS", o_y_pred_dcs))

            o_y_pred_des = des(i_preprocessed_X_train,
                               i_preprocessed_X_test, i_y_train,
                               i_train_base_predictions,
                               i_test_base_predictions,
                               i_train_base_confidences,
                               mt.confs_per_class_per_predictor)
            res_values.append(("DES_Classifier", o_y_pred_des))

            # -- Get Baseline Results
            o_y_pred_vba, _, sel_ind_vba_set = dcs_vba(i_test_base_predictions, i_y_test,
                                                       i_test_base_confidences,
                                                       mt.confs_per_class_per_predictor)
            res_values.append(("DCS_VBA", o_y_pred_vba))

            o_y_pred_sba, _ = dcs_sba(i_preprocessed_X_test, i_y_train, i_train_base_predictions,
                                      i_test_base_predictions, score_metric, score_metric_maximize)
            res_values.append(("DCS_SBA", o_y_pred_sba))

            o_y_pred_vbe = vbe_wrapper(i_preprocessed_X_test, i_y_test, i_test_base_predictions,
                                       i_test_base_confidences, mt.confs_per_class_per_predictor)
            res_values.append(("VBE", o_y_pred_vbe))

            # -- Add meta info to data
            meta_values = []
            meta_values.append(("ground_truth", i_y_test))
            meta_values.append(("Index-Metatask", i_y_test.index))
            meta_values.append(("Fold", [idx for _ in range(len(o_y_pred_vba))]))
            res_values = meta_values + res_values

            if idx == 0:
                # First Fold for this dataset, need to build file
                res_cols = [res_tup[0] for res_tup in res_values]
                fold_results = get_fold_results(res_values)
                res_df = pd.DataFrame(fold_results, columns=res_cols)
                res_df.to_csv(out_path, index=False)
            else:
                fold_results = get_fold_results(res_values)
                res_df = pd.DataFrame(fold_results, columns=res_cols)
                res_df.to_csv(out_path, mode='a', header=False, index=False)

            # Sort final result .csv
            if idx == mt.max_fold:
                tmp_df = pd.read_csv(out_path)
                tmp_df.sort_values(by="Index-Metatask", inplace=True)
                tmp_df.to_csv(out_path, index=False)
