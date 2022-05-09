import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from assembledopenml.metatask import MetaTask
from assembledopenml.compatability.faked_classifier import FakedClassifier
from results.data_utils import get_default_preprocessing

if __name__ == "__main__":
    # --- Input para
    valid_task_ids = [3913, 3560, 9957, 23]
    score_metric = accuracy_score
    score_metric_maximize = True

    # Iterate over tasks to gather results
    nr_tasks = len(valid_task_ids)
    for task_nr, task_id in enumerate(valid_task_ids, start=1):
        mt = MetaTask()
        mt.read_metatask_from_files("../../results/benchmark_metatasks", task_id)
        print("#### Process Task {} for dataset {} ({}/{}) ####".format(mt.openml_task_id, mt.dataset_name,
                                                                        task_nr, nr_tasks))
        pre_proc = get_default_preprocessing()

        # -- Iterate over Folds
        res_cols = None
        max_nr_folds = mt.max_fold + 1  # +1 because folds start counting from 0
        for idx, train_metadata, test_metadata in mt.fold_split(return_fold_index=True):
            print("## Fold {}/{} ##".format(idx + 1, max_nr_folds))

            # Get Data from Metatask
            i_X_train, i_y_train, _, _ = mt.split_meta_dataset(train_metadata)
            i_X_test, i_y_test, i_test_base_predictions, i_test_base_confidences = mt.split_meta_dataset(test_metadata)
            i_X_train = pre_proc.fit_transform(i_X_train)
            i_X_test = pre_proc.transform(i_X_test)

            i_X_meta_train, i_X_meta_test, i_y_meta_train, i_y_meta_test, assert_meta_train_pred, \
            assert_meta_test_pred, assert_meta_train_conf, assert_meta_test_conf = \
                train_test_split(i_X_test, i_y_test, i_test_base_predictions, i_test_base_confidences,
                                 test_size=0.5,
                                 random_state=802349621,
                                 stratify=i_y_test)

            # "How we build the model for usage"
            for i, model_name in enumerate(list(i_test_base_predictions)):
                model_confidences = i_test_base_confidences[["confidence.{}.{}".format(class_name, model_name)
                                                             for class_name in np.unique(i_y_train)]]
                model_predictions = i_test_base_predictions[model_name]

                fc = FakedClassifier(i_X_test, model_predictions, model_confidences)
                fc.fit(i_X_train, i_y_train)
                assert np.array_equal(assert_meta_train_pred[model_name].to_numpy(), fc.predict(i_X_meta_train))
                assert np.array_equal(assert_meta_test_pred[model_name].to_numpy(), fc.predict(i_X_meta_test))
                assert np.array_equal(assert_meta_train_conf[["confidence.{}.{}".format(class_name, model_name)
                                                              for class_name in np.unique(i_y_train)]].to_numpy(),
                                      fc.predict_proba(i_X_meta_train))
                assert np.array_equal(assert_meta_test_conf[["confidence.{}.{}".format(class_name, model_name)
                                                             for class_name in np.unique(i_y_train)]].to_numpy(),
                                      fc.predict_proba(i_X_meta_test))

                # TODO would need to write own check as data checks need specific fake data during init time
                # check_estimator(FakedClassifier(i_X_test, i_y_train, model_predictions, model_confidences))
