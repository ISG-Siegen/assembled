"""

    TODO detailed title

"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

from assembled.metatask import MetaTask
from assembledopenml.openml_assembler import init_dataset_from_task

from results.data_utils import get_default_preprocessing


def get_bm_data(metatask, base_model):
    test_confidences = []
    test_predictions = []
    original_indices = []
    all_oof_data = []

    for fold_idx, X_train, X_test, y_train in mt._exp_yield_base_model_data_across_folds():
        # Get classes because not all bases models have this
        classes_ = np.unique(y_train)

        # Da Basic Preprocessing
        X_train = preproc.fit_transform(X_train)
        X_test = preproc.transform(X_test)
        train_ind, test_ind = mt.get_indices_for_fold(fold_idx, return_indices=True)

        # Get OOF Data (inner validation data)
        oof_confidences = cross_val_predict(bm, X_train, y_train, cv=5,
                                            method="predict_proba")
        oof_predictions = classes_.take(np.argmax(oof_confidences, axis=1), axis=0)
        oof_indices = list(train_ind)
        oof_data = [oof_predictions, oof_confidences, oof_indices]
        all_oof_data.append(oof_data)

        # Get Test Data
        bm.fit(X_train, y_train)
        fold_test_confidences = bm.predict_proba(X_test)
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

    return test_predictions, test_confidences,all_oof_data, classes_


if __name__ == "__main__":

    mt = MetaTask()
    init_dataset_from_task(mt, 3)

    preproc = get_default_preprocessing()

    base_models = [
        ("RF_4", RandomForestClassifier(n_estimators=4)),
        ("RF_5", RandomForestClassifier(n_estimators=5)),
        ("RF_6", RandomForestClassifier(n_estimators=6)),
        ("RF_7", RandomForestClassifier(n_estimators=7)),
    ]

    for bm_name, bm in base_models:
        bm_predictions, bm_confidences, bm_validation_data, bm_classes = get_bm_data(mt, bm)

        # Sort data

        mt.add_predictor(bm_name, bm_predictions, confidences=bm_confidences, conf_class_labels=list(bm_classes),
                         predictor_description=str(bm))

    print()

exit()
