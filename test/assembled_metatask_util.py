import numpy as np

from assembled.metatask import MetaTask

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.datasets import load_breast_cancer, load_iris, load_digits, load_wine
from sklearn.dummy import DummyClassifier


def build_test_classification_metatask(sklearn_toy_dataset_func, id_to_use=-1):
    # Load a Dataset as Dataframe
    task_data = sklearn_toy_dataset_func(as_frame=True)
    target_name = task_data.target.name
    dataset_frame = task_data.frame
    class_labels = np.array([str(x) for x in task_data.target_names])
    feature_names = task_data.feature_names
    cat_feature_names = []
    # Make target column equal class labels again (inverse of label encoder)
    dataset_frame[target_name] = class_labels[dataset_frame[target_name].to_numpy()]

    # Start Filling Metatask with task data
    mt = MetaTask()

    # Get Splits
    fold_indicators = np.empty(len(dataset_frame))
    cv_spliter = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for fold_idx, (train_index, test_index) in enumerate(
            cv_spliter.split(dataset_frame[feature_names], dataset_frame[target_name])):
        fold_indicators[test_index] = fold_idx

    mt.init_dataset_information(dataset_frame, target_name=target_name, class_labels=class_labels,
                                feature_names=feature_names, cat_feature_names=cat_feature_names,
                                task_type="classification", openml_task_id=id_to_use,
                                dataset_name="breast_cancer", folds_indicator=fold_indicators)

    # Get Cross-val-predictions for some base models
    base_models = [
        ("BM1", DummyClassifier(strategy="uniform")),
        ("BM2", DummyClassifier(strategy="most_frequent")),
        ("BM3", DummyClassifier(strategy="prior")),
        ("BM4", DummyClassifier(strategy="uniform")),
        ("BM5", DummyClassifier(strategy="uniform")),
    ]

    for bm_name, bm in base_models:
        classes = np.unique(dataset_frame[target_name])  # because cross_val_predict does not return .classes_
        bm_confidence = cross_val_predict(bm, dataset_frame[feature_names], dataset_frame[target_name], cv=cv_spliter,
                                          n_jobs=1, method="predict_proba")
        bm_predictions = classes.take(np.argmax(bm_confidence, axis=1), axis=0)  # Get original class names instead int

        mt.add_predictor(bm_name, bm_predictions, confidences=bm_confidence, conf_class_labels=list(classes),
                         predictor_description=str(bm))

    # Add Selection Constraints
    mt.read_selection_constraints({"manual": True})

    return mt


def build_multiple_test_classification_metatasks():
    meta_tasks = []
    for idx, load_func in enumerate([load_breast_cancer, load_iris, load_digits, load_wine], 1):
        mt = build_test_classification_metatask(load_func, -idx)
        meta_tasks.append(mt)

    return meta_tasks

# -- Get MEtatasks from openml for tests
# from assembledopenml.openml_assembler import OpenMLAssembler
# from results.data_utils import get_default_preprocessing
#
# import numpy as np
#
# # --- Get Metatasks used for testing
# omla = OpenMLAssembler(openml_metric_name="area_under_roc_curve", maximize_metric=True, nr_base_models=5)
# metatasks = []
# for task_id in [3913, 3560, 9957, 23]:
#     # Build meta-dataset for each task
#     meta_task = omla.run(task_id)
#     meta_task.filter_predictors()
#     metatasks.append(meta_task)
