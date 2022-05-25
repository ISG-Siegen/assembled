"""
    Experimental Code to create metatask from Auto-sklearn (i.e., from its search results)
"""

import os
import glob
import numpy as np
import shutil
import warnings
import copy
from joblib import load
import hashlib

from sklearn.model_selection import StratifiedKFold

import autosklearn.classification
from autosklearn.evaluation.splitter import CustomStratifiedKFold
from autosklearn.evaluation.abstract_evaluator import MyDummyClassifier

from assembled.metatask import MetaTask
from assembledopenml.openml_assembler import init_dataset_from_task
from assembledopenml.compatibility.openml_metrics import OpenMLAUROC

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


def get_bm_data_from_ask(metatask, tmp_folder_name="tmp_model_data"):
    print("Start ASK")
    verify_metric = OpenMLAUROC()

    for fold_idx, X_train, X_test, y_train, y_test in metatask._exp_yield_data_for_base_model_across_folds():
        print("### Processing Fold {}/{} ###".format(fold_idx + 1, metatask.max_fold + 1))

        # Verify clean env
        if os.path.isdir(tmp_folder_name):
            raise ValueError("tmp_folder already exists. We wont delete it. Make sure to delete it yourself.")

        # Fold data
        classes_, y_train_int = np.unique(y_train, return_inverse=True)
        train_ind, _ = metatask.get_indices_for_fold(fold_idx, return_indices=True)

        try:
            automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60 * 60 * 60,
                                                                      tmp_folder=tmp_folder_name,
                                                                      delete_tmp_folder_after_terminate=False,
                                                                      ensemble_size=1,
                                                                      initial_configurations_via_metalearning=1,
                                                                      smac_scenario_args={'runcount_limit': 4},
                                                                      resampling_strategy='cv',
                                                                      resampling_strategy_arguments={'folds': 5},
                                                                      load_models=False)
            automl.fit(X_train.copy(), y_train.copy())
            del automl

            # --- Verify that everything worked and the models dir was build
            if not os.path.isdir(os.path.join(tmp_folder_name, ".auto-sklearn")):
                raise ValueError("tmp_folder is missing for some reason.")

            # --- Verify that we know the correct way the data was split
            true_order_ensemble_y = np.argsort(reproduce_ask_split(y_train, 5))
            ensemble_y = np.load(os.path.join(tmp_folder_name, ".auto-sklearn/true_targets_ensemble.npy"))
            if not np.array_equal(np.unique(y_train, return_inverse=True)[1], ensemble_y[true_order_ensemble_y]):
                raise ValueError("Something wrong with the validation data!")

            # --- For each evaluated base model, store the predictions
            bm_dir_names = [os.path.basename(run_folder) for run_folder in
                            glob.glob(os.path.join("tmp_model_data", ".auto-sklearn/runs/*"))]
            for idx, run_name in enumerate(bm_dir_names):
                print("## Processing Base Model {}/{} ##".format(idx + 1, len(bm_dir_names)))
                path_to_predictions = os.path.join(tmp_folder_name,
                                                   ".auto-sklearn/runs/{0}/predictions_ensemble_{0}.npy".format(
                                                       run_name))

                # -- Get OOF Data
                loaded_confs = np.load(path_to_predictions, allow_pickle=True)
                loaded_preds = np.argmax(loaded_confs, axis=1)
                oof_confs = loaded_confs[true_order_ensemble_y]
                oof_preds = classes_.take(np.argmax(oof_confs, axis=1), axis=0)
                oof_data = (fold_idx, oof_preds, oof_confs, list(train_ind))
                # - Verify performance to be identical
                if verify_metric(ensemble_y, loaded_preds) != verify_metric(y_train, oof_preds):
                    raise ValueError("Something went wrong with loading the files and ordering them correctly.")

                # -- Get Test Data
                clf = load(os.path.join(tmp_folder_name,
                                        ".auto-sklearn/runs/{0}/{1}.model".format(run_name, run_name.replace("_",
                                                                                                             "."))))
                test_confs = clf.fit(X_train.copy(), y_train_int.copy()).predict_proba(X_test)
                test_preds = classes_.take(np.argmax(test_confs, axis=1), axis=0)

                # As timestamp of a model, we take the point in time where its validation predictions were saved.
                # Not the start or stop time internally of AutoML, but the I/O time
                time_end = os.path.getmtime(path_to_predictions)
                dsp_dict = {"time_end": time_end}

                if isinstance(clf, MyDummyClassifier):
                    # Special case because config is empty
                    dsp_dict["configspace"] = str(clf)
                    base_model_dsp = dsp_dict
                    check_sum_for_name = hashlib.md5(str(clf.config).encode('utf-8')).hexdigest()
                    base_model_name = "DummyClassifier" + "({})".format(str(check_sum_for_name))
                else:
                    dsp_dict["configspace"] = clf.config._values
                    base_model_dsp = dsp_dict
                    check_sum_for_name = hashlib.md5(str(clf.config).encode('utf-8')).hexdigest()
                    base_model_name = clf.config["classifier:__choice__"] + "({})".format(str(check_sum_for_name))

                metatask.add_predictor(base_model_name, test_preds, confidences=test_confs,
                                       conf_class_labels=list(classes_), predictor_description=base_model_dsp,
                                       validation_data=[oof_data],
                                       fold_predictor=True, fold_predictor_idx=fold_idx)

        finally:
            # clean up
            if os.path.isdir(tmp_folder_name):
                shutil.rmtree(tmp_folder_name)


def get_openml_metatask_for_ask(mt_id):
    print("Get OpenML Metatask")

    metatask = MetaTask()
    init_dataset_from_task(metatask, mt_id)
    metatask.read_randomness("OpenML", 0)

    return metatask


def get_example_manual_metatask_for_ask():
    print("Get Toy Metatask")
    from sklearn.datasets import load_breast_cancer
    metatask_id = -1
    task_data = load_breast_cancer(as_frame=True)
    target_name = task_data.target.name
    dataset_frame = task_data.frame
    class_labels = np.array([str(x) for x in task_data.target_names])  # cast labels to string
    feature_names = task_data.feature_names
    cat_feature_names = []
    dataset_frame[target_name] = class_labels[dataset_frame[target_name].to_numpy()]
    metatask = MetaTask()
    fold_indicators = np.empty(len(dataset_frame))
    cv_spliter = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for fold_idx, (train_index, test_index) in enumerate(
            cv_spliter.split(dataset_frame[feature_names], dataset_frame[target_name])):
        # This indicates the test subset for fold with number fold_idx
        fold_indicators[test_index] = fold_idx

    metatask.init_dataset_information(dataset_frame, target_name=target_name, class_labels=class_labels,
                                      feature_names=feature_names, cat_feature_names=cat_feature_names,
                                      task_type="classification", openml_task_id=metatask_id,
                                      dataset_name=task_data.filename, folds_indicator=fold_indicators
                                      )
    return metatask


if __name__ == "__main__":
    # Build Metatask and fill dataset
    # mt = get_openml_metatask_for_ask("3")
    mt = get_example_manual_metatask_for_ask()

    bm_data = get_bm_data_from_ask(mt)

    mt.to_files()
