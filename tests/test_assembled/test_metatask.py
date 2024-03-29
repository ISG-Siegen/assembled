from assembled.metatask import MetaTask
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score
from pathlib import Path
from assembledopenml.openml_assembler import init_dataset_from_task

import numpy as np
from tests.assembled_metatask_util import build_multiple_test_classification_metatasks, \
    build_metatask_with_validation_data_with_different_base_models_per_fold, \
    build_metatask_with_validation_data_same_base_models_all_folds, \
    delete_metatask_files

metatasks = build_multiple_test_classification_metatasks()


class TestMetaTask:
    base_path = Path(__file__).parent.resolve()

    def test_metatask_init_data_and_predictor_and_remove_predictor(self):
        # Load a Dataset as Dataframe
        task_data = load_breast_cancer(as_frame=True)
        target_name = task_data.target.name
        dataset_frame = task_data.frame
        class_labels = task_data.target_names
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

        # Test if data init works
        mt.init_dataset_information(dataset_frame, target_name=target_name, class_labels=class_labels,
                                    feature_names=feature_names, cat_feature_names=cat_feature_names,
                                    task_type="classification", openml_task_id=-1,
                                    dataset_name="breast_cancer", folds_indicator=fold_indicators)

        # Test if Predictor is added correctly
        bm = DummyClassifier(strategy="uniform")
        classes = np.unique(dataset_frame[target_name])
        bm_confidence = cross_val_predict(bm, dataset_frame[feature_names], dataset_frame[target_name],
                                          cv=cv_spliter,
                                          n_jobs=1, method="predict_proba")
        bm_predictions = classes.take(np.argmax(bm_confidence, axis=1), axis=0)

        mt.add_predictor("Dummy", bm_predictions, confidences=bm_confidence, conf_class_labels=list(classes),
                         predictor_description=str(bm), bad_predictor=True, corruptions_details={"confs_bad": True})

        conf_cols = [mt.to_confidence_name("Dummy", n) for n in list(classes)]

        assert "Dummy" in mt.predictors
        assert "Dummy" in list(mt.predictions_and_confidences)
        assert all([x in list(mt.predictions_and_confidences) for x in conf_cols])
        assert all([x in mt.confidences for x in conf_cols])
        assert mt.predictor_descriptions["Dummy"] == str(bm)
        assert "Dummy" in mt.bad_predictors
        assert mt.predictor_corruptions_details["Dummy"] == {"confs_bad": True}

        mt.remove_predictors(["Dummy"])

        assert "Dummy" not in mt.predictors
        assert "Dummy" not in list(mt.predictions_and_confidences)
        assert all([x not in list(mt.predictions_and_confidences) for x in conf_cols])
        assert all([x not in mt.confidences for x in conf_cols])
        assert "Dummy" not in list(mt.predictor_descriptions.keys())
        assert "Dummy" not in mt.bad_predictors
        assert "Dummy" not in list(mt.predictor_corruptions_details.keys())

    def test_metatask_fold_split(self):

        for mt in metatasks:

            for idx, train_data, test_data in mt.fold_split(return_fold_index=True):
                test_indices = mt.folds == idx
                train_indices = mt.folds != idx

                assert mt.meta_dataset.iloc[train_indices].equals(train_data)
                assert mt.meta_dataset.iloc[test_indices].equals(test_data)

    def test_metatask_predictor_remove_with_validation_data(self):
        # Load a Dataset as Dataframe
        task_data = load_breast_cancer(as_frame=True)
        target_name = task_data.target.name
        dataset_frame = task_data.frame
        class_labels = task_data.target_names
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

        # Test if data init works
        mt.init_dataset_information(dataset_frame, target_name=target_name, class_labels=class_labels,
                                    feature_names=feature_names, cat_feature_names=cat_feature_names,
                                    task_type="classification", openml_task_id=-1,
                                    dataset_name="breast_cancer", folds_indicator=fold_indicators)

        # Test if Predictor is added correctly
        bm = DummyClassifier(strategy="uniform")

        test_confidences = []
        test_predictions = []
        original_indices = []
        all_oof_data = []

        for fold_idx, X_train, X_test, y_train, _ in mt._exp_yield_data_for_base_model_across_folds():
            # Get classes because not all bases models have this
            classes_ = np.unique(y_train)

            # Da Basic Preprocessing
            train_ind, test_ind = mt.get_indices_for_fold(fold_idx, return_indices=True)

            # Get OOF Data (inner validation data)
            oof_confidences = cross_val_predict(bm, X_train, y_train, cv=5,
                                                method="predict_proba")
            oof_predictions = classes_.take(np.argmax(oof_confidences, axis=1), axis=0)
            oof_indices = list(train_ind)
            oof_data = [fold_idx, oof_predictions, oof_confidences, oof_indices]
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
        bm_confidence = np.array(test_confidences)
        bm_predictions = np.array(test_predictions)

        mt.add_predictor("Dummy", bm_predictions, confidences=bm_confidence, conf_class_labels=list(classes_),
                         predictor_description=str(bm), bad_predictor=True, corruptions_details={"confs_bad": True},
                         validation_data=all_oof_data)

        val_col_preds = [mt.to_validation_predictor_name("Dummy", i) for i in range(10)]
        val_col_confs = [mt.to_confidence_name(p_name, n) for n in
                         mt.class_labels for p_name in val_col_preds]

        assert mt.use_validation_data
        assert all([x in mt.validation_predictions_columns for x in val_col_preds])
        assert all([x in mt.validation_predictions_and_confidences for x in val_col_preds])
        assert all([x in mt.validation_confidences_columns for x in val_col_confs])
        assert all([x in mt.validation_predictions_and_confidences for x in val_col_confs])
        assert mt.split_meta_dataset(mt.meta_dataset)[-2].shape == (569, 10)
        assert mt.split_meta_dataset(mt.meta_dataset)[-1].shape == (569, 20)

        mt.remove_predictors(["Dummy"])

        assert not mt.use_validation_data
        assert all([x not in mt.validation_predictions_columns for x in val_col_preds])
        assert all([x not in mt.validation_predictions_and_confidences for x in val_col_preds])
        assert all([x not in mt.validation_confidences_columns for x in val_col_confs])
        assert all([x not in mt.validation_predictions_and_confidences for x in val_col_confs])
        assert mt.split_meta_dataset(mt.meta_dataset)[-2].shape == (569, 0)
        assert mt.split_meta_dataset(mt.meta_dataset)[-1].shape == (569, 0)

    def test_metatask_filter_predictors(self):

        mt, _, _ = build_metatask_with_validation_data_with_different_base_models_per_fold()

        mt.filter_predictors(max_number_predictors=0, score_metric=balanced_accuracy_score, maximize_metric=True)
        assert mt.fold_predictors == []
        assert all(x == 0 for x in mt.n_predictors_per_fold)

        mt, _ = build_metatask_with_validation_data_same_base_models_all_folds()
        mt.filter_predictors(max_number_predictors=2, score_metric=balanced_accuracy_score, maximize_metric=True)

        assert len(mt.predictors) == 2
        assert mt.predictors == ['RF_6', 'RF_7']

    def test_memory_usage(self):
        mt, _, _ = build_metatask_with_validation_data_with_different_base_models_per_fold(sparse=False, fake_id=-11)
        assert (mt.meta_dataset.memory_usage().sum() / 1e3) == 333.321

        sparse_mt, _, _ = build_metatask_with_validation_data_with_different_base_models_per_fold(sparse=True,
                                                                                                  fake_id=-10)

        assert (sparse_mt.meta_dataset.memory_usage().sum() / 1e3) == 287.801

        # Test write / load
        mt.to_files(self.base_path / "example_metatasks")
        sparse_mt.to_files(self.base_path / "example_metatasks")

        mt = MetaTask()
        mt.read_metatask_from_files(self.base_path / "example_metatasks", -11)
        sprase_mt = MetaTask(use_sparse_dtype=True)
        sprase_mt.read_metatask_from_files(self.base_path / "example_metatasks", -10)

        assert (mt.meta_dataset.memory_usage().sum() / 1e3) == 333.321
        assert (sprase_mt.meta_dataset.memory_usage().sum() / 1e3) == 287.801

    def test_default_file_format(self):
        mt, _, _ = build_metatask_with_validation_data_with_different_base_models_per_fold(fake_id=-12, sparse=False)
        mt.to_files(self.base_path / "example_metatasks")

        r_mt = MetaTask()
        r_mt.read_metatask_from_files(self.base_path / "example_metatasks", -12)

        assert r_mt == mt
        delete_metatask_files(self.base_path / "example_metatasks", -12)

        mt, _, _ = build_metatask_with_validation_data_with_different_base_models_per_fold(fake_id=-12, sparse=True)
        mt.to_files(self.base_path / "example_metatasks")

        r_mt = MetaTask()
        r_mt.read_metatask_from_files(self.base_path / "example_metatasks", -12)

        assert r_mt == mt
        delete_metatask_files(self.base_path / "example_metatasks", -12)

    def test_custom_file_formats(self):

        for file_format in ["hdf", "feather"]:
            print(file_format)
            mt, _, _ = build_metatask_with_validation_data_with_different_base_models_per_fold(fake_id=-12,
                                                                                               file_format=file_format,
                                                                                               sparse=False)
            mt.to_files(self.base_path / "example_metatasks")

            r_mt = MetaTask()
            r_mt.read_metatask_from_files(self.base_path / "example_metatasks", -12)

            assert r_mt == mt
            delete_metatask_files(self.base_path / "example_metatasks", -12, file_format=file_format)

            mt, _, _ = build_metatask_with_validation_data_with_different_base_models_per_fold(fake_id=-12,
                                                                                               file_format=file_format,
                                                                                               sparse=True)
            mt.to_files(self.base_path / "example_metatasks")

            r_mt = MetaTask()
            r_mt.read_metatask_from_files(self.base_path / "example_metatasks", -12)

            assert r_mt == mt
            delete_metatask_files(self.base_path / "example_metatasks", -12, file_format=file_format)

    def test_hdf_with_many_columns_dataset(self):
        metatask = MetaTask()
        init_dataset_from_task(metatask, 10090)
        metatask.file_format = "hdf"
        metatask.to_files(self.base_path / "example_metatasks")

        r_mt = MetaTask()
        r_mt.read_metatask_from_files(self.base_path / "example_metatasks", 10090)

        assert r_mt == metatask

        delete_metatask_files(self.base_path / "example_metatasks", 10090, file_format="hdf")
