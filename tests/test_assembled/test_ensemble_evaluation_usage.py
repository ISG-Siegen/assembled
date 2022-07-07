"""
    An example on how to add validation (e.g.: inner cv) data manually and use it afterwards.
"""
import os
import json
import filecmp

import numpy as np
from pathlib import Path

from ensemble_techniques.util.metrics import OpenMLAUROC
from ensemble_techniques.autosklearn.ensemble_selection import EnsembleSelection

from results.data_utils import get_default_preprocessing
from tests.assembled_metatask_util import build_metatask_with_validation_data_with_different_base_models_per_fold, \
    build_metatask_with_validation_data_same_base_models_all_folds
from assembled.ensemble_evaluation import evaluate_ensemble_on_metatask
from assembled.utils.data_mgmt import merge_fold_results


def test_evaluation_with_validation_data():
    random_base_seed_models = 1
    mt, expected_perf = build_metatask_with_validation_data_same_base_models_all_folds()

    # Example on how to evaluate base models with a metatask that includes validation data
    technique_run_args = {"ensemble_size": 1,
                          "metric": OpenMLAUROC,
                          "random_state": np.random.RandomState(random_base_seed_models)
                          }

    fold_scores = evaluate_ensemble_on_metatask(mt, EnsembleSelection, technique_run_args,
                                                "autosklearn.EnsembleSelection",
                                                pre_fit_base_models=True, preprocessor=get_default_preprocessing(),
                                                use_validation_data_to_train_ensemble_techniques=True,
                                                return_scores=OpenMLAUROC)

    # With the setup as it is (ensemble_size=1), the evaluation will select one of the base models per fold and create
    # the best selection.
    np.testing.assert_array_equal(fold_scores, expected_perf)


def test_evaluation_without_preprocessor():
    random_base_seed_models = 1
    mt, expected_perf = build_metatask_with_validation_data_same_base_models_all_folds()

    # Make a value nan such that nan checks must be performed / ignored.
    mt.dataset.iloc[3, 4] = np.nan
    mt.dataset.iloc[3, 5] = np.nan
    mt.dataset.iloc[:, 5] = mt.dataset.iloc[:, 5].astype("category")

    # Example on how to evaluate base models with a metatask that includes validation data
    technique_run_args = {"ensemble_size": 1,
                          "metric": OpenMLAUROC,
                          "random_state": np.random.RandomState(random_base_seed_models)
                          }

    fold_scores = evaluate_ensemble_on_metatask(mt, EnsembleSelection, technique_run_args,
                                                "autosklearn.EnsembleSelection",
                                                pre_fit_base_models=True, preprocessor=None,
                                                use_validation_data_to_train_ensemble_techniques=True,
                                                return_scores=OpenMLAUROC)

    # With the setup as it is (ensemble_size=1), the evaluation will select one of the base models per fold and create
    # the best selection.
    np.testing.assert_array_equal(fold_scores, expected_perf)


def test_evaluation_with_validation_data_with_different_base_models_per_fold():
    mt, perf_per_fold, perf_per_fold_full = build_metatask_with_validation_data_with_different_base_models_per_fold()

    # -- Eval Test
    technique_run_args = {"ensemble_size": 1,
                          "metric": OpenMLAUROC,
                          "random_state": np.random.RandomState(1)
                          }

    # Without validation data test
    fold_scores = evaluate_ensemble_on_metatask(mt, EnsembleSelection, technique_run_args,
                                                "autosklearn.EnsembleSelection",
                                                pre_fit_base_models=True, preprocessor=get_default_preprocessing(),
                                                return_scores=OpenMLAUROC, meta_train_test_split_random_state=0,
                                                meta_train_test_split_fraction=0.5)

    np.testing.assert_array_equal(fold_scores, perf_per_fold)

    # Validation data test
    fold_scores = evaluate_ensemble_on_metatask(mt, EnsembleSelection, technique_run_args,
                                                "autosklearn.EnsembleSelection",
                                                pre_fit_base_models=True, preprocessor=get_default_preprocessing(),
                                                use_validation_data_to_train_ensemble_techniques=True,
                                                return_scores=OpenMLAUROC)

    np.testing.assert_array_equal(fold_scores, perf_per_fold_full)


def test_evaluation_with_holdout_validation_data():
    random_base_seed_models = 1
    mt, expected_perf = build_metatask_with_validation_data_same_base_models_all_folds(cross_val=False,
                                                                                       expected_ind=[1, 2, 2, 3, 1, 3,
                                                                                                     2, 1, 1, 3])

    # Example on how to evaluate base models with a metatask that includes validation data
    technique_run_args = {"ensemble_size": 1,
                          "metric": OpenMLAUROC,
                          "random_state": np.random.RandomState(random_base_seed_models)
                          }

    fold_scores = evaluate_ensemble_on_metatask(mt, EnsembleSelection, technique_run_args,
                                                "autosklearn.EnsembleSelection",
                                                pre_fit_base_models=True, preprocessor=get_default_preprocessing(),
                                                use_validation_data_to_train_ensemble_techniques=True,
                                                return_scores=OpenMLAUROC)

    # With the setup as it is (ensemble_size=1), the evaluation will select one of the base models per fold and create
    # the best selection.
    np.testing.assert_array_equal(fold_scores, expected_perf)


def test_ensemble_evaluation_file_output():
    # Paths
    base_path = Path(__file__).parent.resolve()
    path_to_known_output = str(base_path.joinpath("known_output_format/known_results_for_metatask_-1.csv"))
    path_to_create_output = str(base_path.joinpath("results_for_metatask_-1.csv"))
    path_to_create_metadata = str(base_path.joinpath("evaluation_metadata_for_metatask_-1.json"))
    base_path = str(base_path)

    # Ensemble and Metatask stuff
    mt, expected_perf = build_metatask_with_validation_data_same_base_models_all_folds()
    technique_run_args = {"ensemble_size": 1, "metric": OpenMLAUROC, "random_state": np.random.RandomState(1)}

    # --- Test Save Sequentially
    evaluate_ensemble_on_metatask(mt, EnsembleSelection, technique_run_args,
                                  "autosklearn.EnsembleSelection1",
                                  pre_fit_base_models=True, preprocessor=get_default_preprocessing(),
                                  use_validation_data_to_train_ensemble_techniques=True,
                                  return_scores=OpenMLAUROC,
                                  output_dir_path=base_path,
                                  save_evaluation_metadata=True,
                                  store_results="sequential")

    evaluate_ensemble_on_metatask(mt, EnsembleSelection, technique_run_args,
                                  "autosklearn.EnsembleSelection2",
                                  pre_fit_base_models=True, preprocessor=get_default_preprocessing(),
                                  use_validation_data_to_train_ensemble_techniques=True,
                                  return_scores=OpenMLAUROC,
                                  output_dir_path=base_path,
                                  save_evaluation_metadata=True,
                                  store_results="sequential")

    evaluate_ensemble_on_metatask(mt, EnsembleSelection, technique_run_args,
                                  "autosklearn.EnsembleSelection3",
                                  pre_fit_base_models=True, preprocessor=get_default_preprocessing(),
                                  use_validation_data_to_train_ensemble_techniques=True,
                                  return_scores=OpenMLAUROC,
                                  output_dir_path=base_path,
                                  save_evaluation_metadata=True,
                                  store_results="sequential")

    # --- Assert Sequential Results
    assert filecmp.cmp(path_to_known_output, path_to_create_output)
    os.remove(path_to_create_output)

    # --- Assert Sequential Metadata
    _verify_metadata(path_to_create_metadata, ["autosklearn.EnsembleSelection1", "autosklearn.EnsembleSelection2",
                                               "autosklearn.EnsembleSelection3"], mt.max_fold + 1)
    os.remove(path_to_create_metadata)

    # --- Test Save Parallel
    evaluate_ensemble_on_metatask(mt, EnsembleSelection, technique_run_args,
                                  "autosklearn.EnsembleSelection1",
                                  pre_fit_base_models=True, preprocessor=get_default_preprocessing(),
                                  use_validation_data_to_train_ensemble_techniques=True,
                                  return_scores=OpenMLAUROC,
                                  output_dir_path=base_path,
                                  save_evaluation_metadata=True,
                                  store_results="parallel")

    evaluate_ensemble_on_metatask(mt, EnsembleSelection, technique_run_args,
                                  "autosklearn.EnsembleSelection2",
                                  pre_fit_base_models=True, preprocessor=get_default_preprocessing(),
                                  use_validation_data_to_train_ensemble_techniques=True,
                                  return_scores=OpenMLAUROC,
                                  output_dir_path=base_path,
                                  save_evaluation_metadata=True,
                                  store_results="parallel")

    evaluate_ensemble_on_metatask(mt, EnsembleSelection, technique_run_args,
                                  "autosklearn.EnsembleSelection3",
                                  pre_fit_base_models=True, preprocessor=get_default_preprocessing(),
                                  use_validation_data_to_train_ensemble_techniques=True,
                                  return_scores=OpenMLAUROC,
                                  output_dir_path=base_path,
                                  save_evaluation_metadata=True,
                                  store_results="parallel")

    merge_fold_results(base_path, mt.openml_task_id, mt.is_classification)

    # --- Assert Parallel Results
    assert filecmp.cmp(path_to_known_output, path_to_create_output)
    os.remove(path_to_create_output)

    # --- Assert Parallel Metadata
    _verify_metadata(path_to_create_metadata, ["autosklearn.EnsembleSelection1", "autosklearn.EnsembleSelection2",
                                               "autosklearn.EnsembleSelection3"], mt.max_fold + 1)
    os.remove(path_to_create_metadata)


def _verify_metadata(path_to_metadata, technique_names, n_folds):
    with open(path_to_metadata, "r") as f:
        eval_md = json.load(f)

    for name in technique_names:
        assert name in eval_md
        assert all(len(md) == n_folds for md in eval_md[name].values())
