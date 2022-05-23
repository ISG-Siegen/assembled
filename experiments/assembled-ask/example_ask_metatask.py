"""
    X
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from assembled.metatask import MetaTask
from assembledopenml.openml_assembler import init_dataset_from_task
from assembledopenml.compatibility.openml_metrics import OpenMLAUROC
from ensemble_techniques.autosklearn.ensemble_selection import EnsembleSelection

from results.data_utils import get_default_preprocessing
import autosklearn.classification
import shutil
from joblib import dump, load
import warnings
import copy

# from autosklearn.evaluation.train_evaluator import
from autosklearn.evaluation.splitter import CustomStratifiedShuffleSplit, CustomStratifiedKFold


def reproduce_ask_split(y, folds):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            cv = StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=1,
            )
            test_cv = copy.deepcopy(cv)
            next(test_cv.split(y, y))
    except UserWarning as e:
        print(e)
        if 'The least populated class in y has only' in e.args[0]:
            cv = CustomStratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=1,
            )
        else:
            raise e

    order_ensemble_y = np.array([j for i in [list(test_idx) for train_idx, test_idx in cv.split(y, y)] for j in i])

    return order_ensemble_y


def get_bm_data_from_ask(metatask, inner_split_random_seed):
    bm_data = []

    for fold_idx, X_train, X_test, y_train, y_test in metatask._exp_yield_data_for_base_model_across_folds():
        shutil.rmtree("tmp_model_data")  # clean up from previous (move to end with finally?)
        automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60 * 60 * 60,
                                                                  tmp_folder="tmp_model_data",
                                                                  delete_tmp_folder_after_terminate=False,
                                                                  ensemble_size=1,
                                                                  initial_configurations_via_metalearning=1,
                                                                  smac_scenario_args={'runcount_limit': 4},
                                                                  resampling_strategy='cv',
                                                                  resampling_strategy_arguments={'folds': 5},
                                                                  )
        automl.fit(X_train.copy(), y_train.copy())

        # --- Verify that we know the correct way the data was split
        true_order_ensemble_y = reproduce_ask_split(y_train, 5)
        ensemble_y = np.load("tmp_model_data/.auto-sklearn/true_targets_ensemble.npy")
        if not np.array_equal(np.unique(y_train, return_inverse=True)[1],
                              ensemble_y[np.argsort(true_order_ensemble_y)]):
            raise ValueError("Something wrong with the validation data!")

        # --- For each evaluated base model, store the predictions
        for run_name in ["1_1_0.0"]:
            oof_confs = np.load("tmp_model_data/.auto-sklearn/runs/{0}/predictions_ensemble_{0}.npy".format(run_name),
                                allow_pickle=True)[true_order_ensemble_y]
            clf = load("tmp_model_data/.auto-sklearn/runs/{0}/{0}.model".format(run_name))
            test_conf = clf.fit(X_train.copy(), y_train.copy()).predict_proba(X_test)

            # TODO get description / name from clf ; get predictions; store in list/object or add to metatask immediately

    # test_confidences = []
    # test_predictions = []
    # original_indices = []
    # all_oof_data = []
    # fold_perfs = []
    #
    # for fold_idx, X_train, X_test, y_train, y_test in metatask._exp_yield_data_for_base_model_across_folds():
    #     # Get classes because not all bases models have this
    #     classes_ = np.unique(y_train)
    #
    #     # Da Basic Preprocessing
    #     X_train = preprocessing.fit_transform(X_train)
    #     X_test = preprocessing.transform(X_test)
    #     train_ind, test_ind = metatask.get_indices_for_fold(fold_idx, return_indices=True)
    #
    #     # Get OOF Data (inner validation data)
    #     oof_confidences = cross_val_predict(base_model, X_train, y_train,
    #                                         cv=StratifiedKFold(n_splits=5, shuffle=True,
    #                                                            random_state=inner_split_random_seed),
    #                                         method="predict_proba")
    #     oof_predictions = classes_.take(np.argmax(oof_confidences, axis=1), axis=0)
    #     oof_indices = list(train_ind)
    #     oof_data = [fold_idx, oof_predictions, oof_confidences, oof_indices]
    #     all_oof_data.append(oof_data)
    #
    #     # Get Test Data
    #     base_model.fit(X_train, y_train)
    #     fold_test_confidences = base_model.predict_proba(X_test)
    #     fold_test_predictions = classes_.take(np.argmax(fold_test_confidences, axis=1), axis=0)
    #     fold_indices = list(test_ind)
    #
    #     # Add to data
    #     test_confidences.extend(fold_test_confidences)
    #     test_predictions.extend(fold_test_predictions)
    #     original_indices.extend(fold_indices)
    #     fold_perfs.append(OpenMLAUROC()(y_test, fold_test_predictions))
    #
    # test_confidences, test_predictions, original_indices = zip(
    #     *sorted(zip(test_confidences, test_predictions, original_indices),
    #             key=lambda x: x[2]))
    # test_confidences = np.array(test_confidences)
    # test_predictions = np.array(test_predictions)
    #
    # return test_predictions, test_confidences, all_oof_data, classes_
    # # bm_predictions, bm_confidences, bm_validation_data, bm_classes, bm_name, bm_description


if __name__ == "__main__":
    # Control Randomness
    random_base_seed_models = 1
    random_base_seed_data = 0
    metatask_id = "3"

    base_rnger = np.random.RandomState(random_base_seed_data)
    random_int_seed_outer_folds = base_rnger.randint(0, 10000000)
    random_int_seed_inner_folds = base_rnger.randint(0, 10000000)

    # Build Metatask and fill dataset
    mt = MetaTask()
    init_dataset_from_task(mt, metatask_id)
    mt.read_randomness(random_int_seed_outer_folds, random_int_seed_inner_folds)

    bm_data = get_bm_data_from_ask(mt, random_int_seed_inner_folds)

    # Sort data
    for bm_predictions, bm_confidences, bm_validation_data, bm_classes, bm_name, bm_description in bm_data:
        mt.add_predictor(bm_name, bm_predictions, confidences=bm_confidences, conf_class_labels=list(bm_classes),
                         predictor_description=bm_description, validation_data=bm_validation_data)

    mt.to_files()
