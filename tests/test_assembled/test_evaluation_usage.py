"""
    An example on how to add validation (e.g.: inner cv) data manually and use it afterwards.
"""
import numpy as np

from ensemble_techniques.util.metrics import OpenMLAUROC
from ensemble_techniques.autosklearn.ensemble_selection import EnsembleSelection

from results.data_utils import get_default_preprocessing
from tests.assembled_metatask_util import build_metatask_with_validation_data_with_different_base_models_per_fold, \
    build_metatask_with_validation_data_same_base_models_all_folds


def test_evaluation_with_validation_data():
    random_base_seed_models = 1
    mt, expected_perf = build_metatask_with_validation_data_same_base_models_all_folds()

    # Example on how to evaluate base models with a metatask that includes validation data
    technique_run_args = {"ensemble_size": 1,
                          "metric": OpenMLAUROC,
                          "random_state": np.random.RandomState(random_base_seed_models)
                          }

    fold_scores = mt.run_ensemble_on_all_folds(EnsembleSelection, technique_run_args, "autosklearn.EnsembleSelection",
                                               pre_fit_base_models=True, preprocessor=get_default_preprocessing(),
                                               use_validation_data_to_train_ensemble_techniques=True,
                                               return_scores=OpenMLAUROC)

    # With the setup as it is (ensemble_size=1), the evaluation will select one of the base models per fold and create
    # the best selection.
    np.testing.assert_array_equal(fold_scores, expected_perf)


def test_evaluation_without_preprocessor():
    random_base_seed_models = 1
    mt, expected_perf = build_metatask_with_validation_data_same_base_models_all_folds()

    # Example on how to evaluate base models with a metatask that includes validation data
    technique_run_args = {"ensemble_size": 1,
                          "metric": OpenMLAUROC,
                          "random_state": np.random.RandomState(random_base_seed_models)
                          }

    fold_scores = mt.run_ensemble_on_all_folds(EnsembleSelection, technique_run_args, "autosklearn.EnsembleSelection",
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
    fold_scores = mt.run_ensemble_on_all_folds(EnsembleSelection, technique_run_args, "autosklearn.EnsembleSelection",
                                               pre_fit_base_models=True, preprocessor=get_default_preprocessing(),
                                               return_scores=OpenMLAUROC, meta_train_test_split_random_state=0,
                                               meta_train_test_split_fraction=0.5)

    np.testing.assert_array_equal(fold_scores, perf_per_fold)

    # Validation data test
    fold_scores = mt.run_ensemble_on_all_folds(EnsembleSelection, technique_run_args, "autosklearn.EnsembleSelection",
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

    fold_scores = mt.run_ensemble_on_all_folds(EnsembleSelection, technique_run_args, "autosklearn.EnsembleSelection",
                                               pre_fit_base_models=True, preprocessor=get_default_preprocessing(),
                                               use_validation_data_to_train_ensemble_techniques=True,
                                               return_scores=OpenMLAUROC)

    # With the setup as it is (ensemble_size=1), the evaluation will select one of the base models per fold and create
    # the best selection.
    np.testing.assert_array_equal(fold_scores, expected_perf)
