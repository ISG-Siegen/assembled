import numpy as np

from assembled.metatask import MetaTask

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.datasets import load_breast_cancer, load_iris, load_digits, load_wine
from sklearn.dummy import DummyClassifier

from ensemble_techniques.util.metrics import OpenMLAUROC
from assembledopenml.openml_assembler import init_dataset_from_task

from results.data_utils import get_default_preprocessing


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


# -- Utils
def get_bm_data(metatask, base_model, preprocessing, inner_split_random_seed, cross_val=True):
    test_confidences = []
    test_predictions = []
    original_indices = []
    all_oof_data = []
    fold_perfs = []
    inner_split_rng = np.random.RandomState(inner_split_random_seed)

    for fold_idx, X_train, X_test, y_train, y_test in metatask._exp_yield_data_for_base_model_across_folds():
        # Get classes because not all bases models have this
        classes_ = np.unique(y_train)

        # Da Basic Preprocessing
        X_train = preprocessing.fit_transform(X_train)
        X_test = preprocessing.transform(X_test)
        train_ind, test_ind = metatask.get_indices_for_fold(fold_idx, return_indices=True)

        # Get OOF Data (inner validation data)
        if cross_val:
            oof_confidences = cross_val_predict(base_model, X_train, y_train,
                                                cv=StratifiedKFold(n_splits=5, shuffle=True,
                                                                   random_state=inner_split_rng),
                                                method="predict_proba")
            oof_predictions = classes_.take(np.argmax(oof_confidences, axis=1), axis=0)
            oof_indices = np.array(list(train_ind))
            oof_data = (fold_idx, oof_predictions, oof_confidences, oof_indices)
        else:
            val_X_train, val_X_test, val_y_train, _, _, val_indices = train_test_split(X_train, y_train, train_ind,
                                                                                       test_size=0.25, stratify=y_train,
                                                                                       random_state=1)
            oof_confidences = base_model.fit(val_X_train, val_y_train).predict_proba(val_X_test)
            oof_predictions = classes_.take(np.argmax(oof_confidences, axis=1), axis=0)
            oof_indices = np.array(list(val_indices))
            oof_data = (fold_idx, oof_predictions, oof_confidences, oof_indices)

        all_oof_data.append(oof_data)

        # Get Test Data
        base_model.fit(X_train, y_train)
        fold_test_confidences = base_model.predict_proba(X_test)
        fold_test_predictions = classes_.take(np.argmax(fold_test_confidences, axis=1), axis=0)
        fold_indices = list(test_ind)

        # Add to data
        test_confidences.extend(fold_test_confidences)
        test_predictions.extend(fold_test_predictions)
        original_indices.extend(fold_indices)
        fold_perfs.append(OpenMLAUROC(y_test, fold_test_predictions))

    test_confidences, test_predictions, original_indices = zip(*sorted(zip(test_confidences, test_predictions,
                                                                           original_indices), key=lambda x: x[2]))
    test_confidences = np.array(test_confidences)
    test_predictions = np.array(test_predictions)

    return test_predictions, test_confidences, all_oof_data, classes_, fold_perfs


def build_metatask_with_validation_data_with_different_base_models_per_fold(sparse=False, fake_id=None, hdf=False):
    # Build Metatask and fill dataset
    task_data = load_breast_cancer(as_frame=True)
    target_name = task_data.target.name
    dataset_frame = task_data.frame
    class_labels = task_data.target_names
    feature_names = task_data.feature_names
    cat_feature_names = []
    dataset_frame[target_name] = class_labels[dataset_frame[target_name].to_numpy()]
    mt = MetaTask(use_sparse_dtype=sparse, use_hdf_file_format=hdf)
    task_id = -1 if fake_id is None else fake_id
    dummy_fold_indicator = np.array([0 for _ in range(len(dataset_frame))])
    mt.init_dataset_information(dataset_frame, target_name=target_name, class_labels=class_labels,
                                feature_names=feature_names, cat_feature_names=cat_feature_names,
                                task_type="classification", openml_task_id=task_id,  # if not an OpenML task, use -X for now
                                dataset_name="breast_cancer", folds_indicator=dummy_fold_indicator)
    new_fold_indicator = np.zeros(mt.n_instances)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for fold_idx, (_, test_indices) in enumerate(cv.split(mt.dataset[mt.feature_names], mt.dataset[mt.target_name])):
        new_fold_indicator[test_indices] = fold_idx
    mt.read_folds(new_fold_indicator)

    base_models_per_fold = [
        ("RF_1", RandomForestClassifier(n_estimators=1, random_state=1)),
        ("RF_2", RandomForestClassifier(n_estimators=2, random_state=2)),
        ("RF_3", RandomForestClassifier(n_estimators=3, random_state=3)),
        ("RF_4", RandomForestClassifier(n_estimators=4, random_state=4)),
        ("RF_5", RandomForestClassifier(n_estimators=5, random_state=5)),
        ("RF_6", RandomForestClassifier(n_estimators=6, random_state=6)),
        ("RF_7", RandomForestClassifier(n_estimators=7, random_state=7)),
        ("RF_8", RandomForestClassifier(n_estimators=8, random_state=8)),
        ("RF_9", RandomForestClassifier(n_estimators=9, random_state=9)),
        ("RF_10", RandomForestClassifier(n_estimators=10, random_state=10)),

    ]

    preproc = get_default_preprocessing()
    perf_per_fold = []
    perf_per_fold_full = []
    for fold_idx, X_train, X_test, y_train, y_test in mt._exp_yield_data_for_base_model_across_folds():
        X_train = preproc.fit_transform(X_train)
        X_test = preproc.transform(X_test)
        base_model = base_models_per_fold[fold_idx][1]
        base_model_name = base_models_per_fold[fold_idx][0]
        base_model_dsp = str(base_model)

        classes_ = np.unique(y_train)

        train_ind, _ = mt.get_indices_for_fold(fold_idx, return_indices=True)

        # OOF Data
        oof_confidences = cross_val_predict(base_model, X_train, y_train,
                                            cv=StratifiedKFold(n_splits=2, shuffle=True,
                                                               random_state=1),
                                            method="predict_proba")
        oof_predictions = classes_.take(np.argmax(oof_confidences, axis=1), axis=0)
        oof_indices = train_ind
        oof_data = (fold_idx, oof_predictions, oof_confidences, oof_indices)

        # Get test data
        base_model.fit(X_train, y_train)
        fold_test_confidences = base_model.predict_proba(X_test)
        fold_test_predictions = classes_.take(np.argmax(fold_test_confidences, axis=1), axis=0)

        # Get perf data for evaluation
        _, y_pred, _, y_true = train_test_split(fold_test_predictions, y_test, test_size=0.5, random_state=0,
                                                stratify=y_test)
        perf_per_fold.append(OpenMLAUROC(y_true, y_pred))
        perf_per_fold_full.append(OpenMLAUROC(y_test, fold_test_predictions))

        # Fill PRedictor
        mt.add_predictor(base_model_name, fold_test_predictions, confidences=fold_test_confidences,
                         conf_class_labels=list(classes_), predictor_description=base_model_dsp,
                         validation_data=[oof_data],
                         fold_predictor=True, fold_predictor_idx=fold_idx)

    return mt, perf_per_fold, perf_per_fold_full


def build_metatask_with_validation_data_same_base_models_all_folds(openml_task=None, expected_ind=None, cross_val=True):
    # Control Randomness
    random_base_seed_data = 0
    base_rnger = np.random.RandomState(random_base_seed_data)
    random_int_seed_outer_folds = base_rnger.randint(0, 10000000)
    random_int_seed_inner_folds = base_rnger.randint(0, 10000000)

    # Build Metatask and fill dataset
    if openml_task is None:
        task_data = load_breast_cancer(as_frame=True)
        target_name = task_data.target.name
        dataset_frame = task_data.frame
        class_labels = task_data.target_names
        feature_names = task_data.feature_names
        cat_feature_names = []
        # Make target column equal class labels again (inverse of label encoder)
        dataset_frame[target_name] = class_labels[dataset_frame[target_name].to_numpy()]
        mt = MetaTask()
        dummy_fold_indicator = np.array([0 for _ in range(len(dataset_frame))])
        mt.init_dataset_information(dataset_frame, target_name=target_name, class_labels=class_labels,
                                    feature_names=feature_names, cat_feature_names=cat_feature_names,
                                    task_type="classification", openml_task_id=-1,
                                    # if not an OpenML task, use -X for now
                                    dataset_name="breast_cancer", folds_indicator=dummy_fold_indicator
                                    )
    else:
        mt = MetaTask()
        init_dataset_from_task(mt, openml_task)
    mt.read_randomness(random_int_seed_outer_folds, random_int_seed_inner_folds)

    # Get custom outer splits (not recommended for OpenML metatasks as it makes it less comparable to results on OpenML)
    # This is here for demonstration purposes.
    new_fold_indicator = np.zeros(mt.n_instances)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_int_seed_outer_folds)
    for fold_idx, (_, test_indices) in enumerate(cv.split(mt.dataset[mt.feature_names], mt.dataset[mt.target_name])):
        new_fold_indicator[test_indices] = fold_idx

    # Fill metataks with new fold indicator and overwrite stuff rom OpenML.
    mt.read_folds(new_fold_indicator)

    preproc = get_default_preprocessing()

    base_models = [
        ("RF_4", RandomForestClassifier(n_estimators=4, random_state=1)),
        ("RF_5", RandomForestClassifier(n_estimators=5, random_state=2)),
        ("RF_6", RandomForestClassifier(n_estimators=6, random_state=3)),
        ("RF_7", RandomForestClassifier(n_estimators=7, random_state=4)),
    ]

    expected_perf = np.empty(10)
    expected_ind = np.array([0, 3, 3, 3, 2, 3, 2, 2, 3, 2]) if expected_ind is None else np.array(expected_ind)
    print()
    for idx, (bm_name, bm) in enumerate(base_models):
        bm_predictions, bm_confidences, bm_validation_data, bm_classes, fold_perfs = get_bm_data(mt, bm, preproc,
                                                                                                 random_int_seed_inner_folds,
                                                                                                 cross_val=cross_val)
        expected_perf[expected_ind == idx] = np.array(fold_perfs)[expected_ind == idx]
        print(fold_perfs)

        # Sort data
        mt.add_predictor(bm_name, bm_predictions, confidences=bm_confidences, conf_class_labels=list(bm_classes),
                         predictor_description=str(bm), validation_data=bm_validation_data)

    return mt, expected_perf
