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
    build_metatask_with_validation_data_same_base_models_all_folds, delete_metatask_files
from assembled.ensemble_evaluation import evaluate_ensemble_on_metatask
from assembled.metatask import MetaTask
from assembled.utils.data_mgmt import merge_fold_results
from ensemble_techniques.util.metrics import make_metric
from sklearn.metrics import roc_auc_score


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


def test_evaluation_random_state():
    from ensemble_techniques.wrapper.abstract_ensemble import AbstractEnsemble
    # A random selection ensemble
    class RandomSelection(AbstractEnsemble):
        """A Random selection wrapper"""

        def __init__(self, base_models, random_state) -> None:

            super().__init__(base_models, "predict_proba", "predict_proba")
            self.random_state = random_state

        def ensemble_fit(self, predictions, labels):
            self.weights_ = np.zeros(len(predictions))
            self.weights_[self.random_state.choice(list(range(len(predictions))))] = 1

            return self

        def ensemble_predict(self, predictions) -> np.ndarray:

            average = np.zeros_like(predictions[0], dtype=np.float64)
            tmp_predictions = np.empty_like(predictions[0], dtype=np.float64)

            # if predictions.shape[0] == len(self.weights_),
            # predictions include those of zero-weight models.
            if len(predictions) == len(self.weights_):
                for pred, weight in zip(predictions, self.weights_):
                    np.multiply(pred, weight, out=tmp_predictions)
                    np.add(average, tmp_predictions, out=average)

            # if prediction model.shape[0] == len(non_null_weights),
            # predictions do not include those of zero-weight models.
            elif len(predictions) == np.count_nonzero(self.weights_):
                non_null_weights = [w for w in self.weights_ if w > 0]
                for pred, weight in zip(predictions, non_null_weights):
                    np.multiply(pred, weight, out=tmp_predictions)
                    np.add(average, tmp_predictions, out=average)

            # If none of the above applies, then something must have gone wrong.
            else:
                raise ValueError("The dimensions of ensemble predictions"
                                 " and ensemble weights do not match!")
            del tmp_predictions
            return average

    # Random selected scores
    mt, expected_perf = build_metatask_with_validation_data_same_base_models_all_folds(
        rng_perf=np.random.RandomState(1))

    # -- Basic Usage
    fold_scores = evaluate_ensemble_on_metatask(mt, RandomSelection, {"random_state": np.random.RandomState(1)},
                                                "autosklearn.EnsembleSelection",
                                                pre_fit_base_models=True, preprocessor=get_default_preprocessing(),
                                                use_validation_data_to_train_ensemble_techniques=True,
                                                return_scores=OpenMLAUROC)
    np.testing.assert_array_equal(fold_scores, expected_perf)

    # -- Isolation usage
    import platform
    if platform.system() == "Windows":
        raise ValueError("You require linux to run this test!")
    fold_scores = evaluate_ensemble_on_metatask(mt, RandomSelection, {"random_state": np.random.RandomState(1)},
                                                "autosklearn.EnsembleSelection",
                                                pre_fit_base_models=True, preprocessor=get_default_preprocessing(),
                                                use_validation_data_to_train_ensemble_techniques=True,
                                                return_scores=OpenMLAUROC, isolate_ensemble_execution=True)

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


def test_evaluation_with_proba_metric():
    random_base_seed_models = 1
    mt, _ = build_metatask_with_validation_data_same_base_models_all_folds()

    metric = make_metric(roc_auc_score,
                         metric_name="roc_auc",
                         maximize=True,
                         classification=True,
                         always_transform_conf_to_pred=False,
                         optimum_value=1,
                         requires_confidences=True,
                         only_positive_class=True,
                         pos_label=1)

    # Example on how to evaluate base models with a metatask that includes validation data
    technique_run_args = {"ensemble_size": 1,
                          "metric": metric,
                          "random_state": np.random.RandomState(random_base_seed_models)
                          }

    fold_scores = evaluate_ensemble_on_metatask(mt, EnsembleSelection, technique_run_args,
                                                "autosklearn.EnsembleSelection",
                                                pre_fit_base_models=True, preprocessor=get_default_preprocessing(),
                                                use_validation_data_to_train_ensemble_techniques=True,
                                                return_scores=metric,
                                                predict_method="predict_proba")

    assert all(np.isclose(fold_scores, np.array([0.993506, 1., 1., 0.997354,
                                                 0.945106, 0.970899, 0.970899, 1., 0.993386, 0.97551])))


def test_ensemble_evaluation_file_output_predict():
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


def test_ensemble_evaluation_file_output_predict_proba():
    # Paths
    base_path = Path(__file__).parent.resolve()
    path_to_known_output = str(
        base_path.joinpath("known_output_format/known_results_predict_proba_for_metatask_-1.csv"))
    path_to_create_output = str(base_path.joinpath("results_for_metatask_-1.csv"))
    path_to_create_metadata = str(base_path.joinpath("evaluation_metadata_for_metatask_-1.json"))
    base_path = str(base_path)

    # Ensemble and Metatask stuff
    metric = make_metric(roc_auc_score,
                         metric_name="roc_auc",
                         maximize=True,
                         classification=True,
                         always_transform_conf_to_pred=False,
                         optimum_value=1,
                         requires_confidences=True,
                         only_positive_class=True,
                         pos_label=1)

    mt, expected_perf = build_metatask_with_validation_data_same_base_models_all_folds()
    technique_run_args = {"ensemble_size": 1, "metric": metric, "random_state": np.random.RandomState(1)}
    eval_args = {"pre_fit_base_models": True, "preprocessor": get_default_preprocessing(),
                 "use_validation_data_to_train_ensemble_techniques": True, "return_scores": metric,
                 "output_dir_path": base_path, "save_evaluation_metadata": True, "predict_method": "predict_proba"}

    # --- Test Save Sequentially
    evaluate_ensemble_on_metatask(mt, EnsembleSelection, technique_run_args,
                                  "autosklearn.EnsembleSelection1", store_results="sequential",
                                  **eval_args)

    evaluate_ensemble_on_metatask(mt, EnsembleSelection, technique_run_args,
                                  "autosklearn.EnsembleSelection2", store_results="sequential",
                                  **eval_args)

    evaluate_ensemble_on_metatask(mt, EnsembleSelection, technique_run_args,
                                  "autosklearn.EnsembleSelection3",
                                  store_results="sequential",
                                  **eval_args)

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
                                  store_results="parallel",
                                  **eval_args)

    evaluate_ensemble_on_metatask(mt, EnsembleSelection, technique_run_args,
                                  "autosklearn.EnsembleSelection2",
                                  store_results="parallel",
                                  **eval_args)

    evaluate_ensemble_on_metatask(mt, EnsembleSelection, technique_run_args,
                                  "autosklearn.EnsembleSelection3",
                                  store_results="parallel",
                                  **eval_args)

    merge_fold_results(base_path, mt.openml_task_id, mt.is_classification, predict_method="predict_proba")

    # --- Assert Parallel Results
    assert filecmp.cmp(path_to_known_output, path_to_create_output)
    os.remove(path_to_create_output)

    # --- Assert Parallel Metadata
    _verify_metadata(path_to_create_metadata, ["autosklearn.EnsembleSelection1", "autosklearn.EnsembleSelection2",
                                               "autosklearn.EnsembleSelection3"], mt.max_fold + 1)
    os.remove(path_to_create_metadata)


def test_evaluation_with_delayed_load():
    random_base_seed_models = 1
    mt, expected_perf = build_metatask_with_validation_data_same_base_models_all_folds(cross_val=False,
                                                                                       expected_ind=[1, 2, 2, 3, 1, 3,
                                                                                                     2, 1, 1, 3])
    mt.file_format = "hdf"
    mt.to_files("./")
    del mt

    mt = MetaTask()
    mt.read_metatask_from_files("./", -1, delayed_evaluation_load=True)

    technique_run_args = {"ensemble_size": 1, "metric": OpenMLAUROC,
                          "random_state": np.random.RandomState(random_base_seed_models)}
    fold_scores = evaluate_ensemble_on_metatask(mt, EnsembleSelection, technique_run_args,
                                                "autosklearn.EnsembleSelection",
                                                pre_fit_base_models=True, preprocessor=get_default_preprocessing(),
                                                use_validation_data_to_train_ensemble_techniques=True,
                                                return_scores=OpenMLAUROC, verbose=True)

    # With the setup as it is (ensemble_size=1), the evaluation will select one of the base models per fold and create
    # the best selection.
    np.testing.assert_array_equal(fold_scores, expected_perf)
    delete_metatask_files("./", -1, file_format=mt.file_format)


def _verify_metadata(path_to_metadata, technique_names, n_folds):
    with open(path_to_metadata, "r") as f:
        eval_md = json.load(f)

    for name in technique_names:
        assert name in eval_md
        assert all(len(md) == n_folds for md in eval_md[name].values())


def test_evaluation_with_passing_metadata_to_base_model():
    from ensemble_techniques.wrapper.abstract_ensemble import AbstractEnsemble
    from assembled.compatibility.faked_classifier import FakedClassifier
    from typing import List
    # A random selection ensemble

    class RandomSelection(AbstractEnsemble):
        """A Random selection wrapper that tries to look at the metadata of base models"""

        def __init__(self, base_models, random_state) -> None:

            super().__init__(base_models, "predict_proba", "predict_proba")
            self.random_state = random_state
            self._inspect_base_model_metadata(base_models)

        def _inspect_base_model_metadata(self, base_models: List[FakedClassifier]):
            for bm in base_models:
                assert hasattr(bm, "model_metadata")
                assert bm.model_metadata is not None

        def ensemble_fit(self, predictions, labels):
            self.weights_ = np.zeros(len(predictions))
            self.weights_[self.random_state.choice(list(range(len(predictions))))] = 1

            return self

        def ensemble_predict(self, predictions) -> np.ndarray:

            average = np.zeros_like(predictions[0], dtype=np.float64)
            tmp_predictions = np.empty_like(predictions[0], dtype=np.float64)

            # if predictions.shape[0] == len(self.weights_),
            # predictions include those of zero-weight models.
            if len(predictions) == len(self.weights_):
                for pred, weight in zip(predictions, self.weights_):
                    np.multiply(pred, weight, out=tmp_predictions)
                    np.add(average, tmp_predictions, out=average)

            # if prediction model.shape[0] == len(non_null_weights),
            # predictions do not include those of zero-weight models.
            elif len(predictions) == np.count_nonzero(self.weights_):
                non_null_weights = [w for w in self.weights_ if w > 0]
                for pred, weight in zip(predictions, non_null_weights):
                    np.multiply(pred, weight, out=tmp_predictions)
                    np.add(average, tmp_predictions, out=average)

            # If none of the above applies, then something must have gone wrong.
            else:
                raise ValueError("The dimensions of ensemble predictions"
                                 " and ensemble weights do not match!")
            del tmp_predictions
            return average

    # Random selected scores
    mt, expected_perf = build_metatask_with_validation_data_same_base_models_all_folds(
        rng_perf=np.random.RandomState(1))

    # -- Basic Usage
    fold_scores = evaluate_ensemble_on_metatask(mt, RandomSelection, {"random_state": np.random.RandomState(1)},
                                                "autosklearn.EnsembleSelection",
                                                pre_fit_base_models=True, preprocessor=get_default_preprocessing(),
                                                use_validation_data_to_train_ensemble_techniques=True,
                                                return_scores=OpenMLAUROC,
                                                store_metadata_in_fake_base_model=True)
    np.testing.assert_array_equal(fold_scores, expected_perf)
