"""
    An example on how to add validation (e.g.: inner cv) data manually and use it afterwards.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from assembled.metatask import MetaTask
from assembledopenml.openml_assembler import init_dataset_from_task

from results.data_utils import get_default_preprocessing


def get_bm_data(metatask, base_model, inner_split_random_seed):
    test_confidences = []
    test_predictions = []
    original_indices = []
    all_oof_data = []
    inner_split_rng = np.random.RandomState(inner_split_random_seed)

    for fold_idx, X_train, X_test, y_train in metatask._exp_yield_data_for_base_model_across_folds():
        # Get classes because not all bases models have this
        classes_ = np.unique(y_train)

        # Da Basic Preprocessing
        X_train = preproc.fit_transform(X_train)
        X_test = preproc.transform(X_test)
        train_ind, test_ind = metatask.get_indices_for_fold(fold_idx, return_indices=True)

        # Get OOF Data (inner validation data)
        oof_confidences = cross_val_predict(base_model, X_train, y_train,
                                            cv=StratifiedKFold(n_splits=5, shuffle=True,
                                                               random_state=inner_split_rng),
                                            method="predict_proba")
        oof_predictions = classes_.take(np.argmax(oof_confidences, axis=1), axis=0)
        oof_indices = list(train_ind)
        oof_data = [fold_idx, oof_predictions, oof_confidences, oof_indices]
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

    test_confidences, test_predictions, original_indices = zip(
        *sorted(zip(test_confidences, test_predictions, original_indices),
                key=lambda x: x[2]))
    test_confidences = np.array(test_confidences)
    test_predictions = np.array(test_predictions)

    return test_predictions, test_confidences, all_oof_data, classes_


if __name__ == "__main__":
    # Control Randomness
    random_base_seed_models = 1
    random_base_seed_data = 0
    base_rnger = np.random.RandomState(random_base_seed_data)
    random_int_seed_outer_folds = base_rnger.randint(0, 10000000)
    random_int_seed_inner_folds = base_rnger.randint(0, 10000000)

    # Build Metatask and fill dataset
    mt = MetaTask()
    init_dataset_from_task(mt, 3)
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
        ("RF_4", RandomForestClassifier(n_estimators=4, random_state=np.random.RandomState(random_base_seed_models))),
        ("RF_5", RandomForestClassifier(n_estimators=5, random_state=np.random.RandomState(random_base_seed_models))),
        ("RF_6", RandomForestClassifier(n_estimators=6, random_state=np.random.RandomState(random_base_seed_models))),
        ("RF_7", RandomForestClassifier(n_estimators=7, random_state=np.random.RandomState(random_base_seed_models))),
    ]

    for bm_name, bm in base_models:
        bm_predictions, bm_confidences, bm_validation_data, bm_classes = get_bm_data(mt, bm,
                                                                                     random_int_seed_inner_folds)

        # Sort data
        mt.add_predictor(bm_name, bm_predictions, confidences=bm_confidences, conf_class_labels=list(bm_classes),
                         predictor_description=str(bm), validation_data=bm_validation_data)
