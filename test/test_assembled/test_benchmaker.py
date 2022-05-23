import pathlib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from assembled.metatask import MetaTask
from assembled.benchmaker import BenchMaker, rebuild_benchmark
from assembledopenml.openml_assembler import OpenMLAssembler, task_to_dataset, init_dataset_from_task
from assembledopenml.compatibility.openml_metrics import OpenMLAUROC
from results.data_utils import get_default_preprocessing


class TestBenchMaker:
    base_path = pathlib.Path(__file__).parent.resolve()

    def test_benchmaker(self):
        base_path = self.base_path
        mt_ids = ["-1", "-2", "-3", "-4"]

        # -- Build Benchmark
        bmer = BenchMaker(base_path / "example_metatasks", base_path / "example_benchmark_metatasks",
                          manual_filter_duplicates=False, min_number_predictors=5,
                          remove_bad_predictors=True, remove_worse_than_random_predictors=True,
                          remove_constant_predictors=True, tasks_to_use=mt_ids,
                          metric_info=(accuracy_score, "acc", True))

        bmer.build_benchmark(share_data="share_meta_data")

        # -- Read Metatasks from the Benchmark

        original_mts = {}
        for mt_id in mt_ids:
            mt = MetaTask()
            mt.read_metatask_from_files(base_path / "example_benchmark_metatasks", mt_id)
            original_mts[mt_id] = mt

        # -- Re-Build benchmark

        def sklearn_load_dataset_function_to_used_data(func):
            task_data = func(as_frame=True)
            dataset_frame = task_data.frame
            class_labels = np.array([str(x) for x in task_data.target_names])
            dataset_frame[task_data.target.name] = class_labels[dataset_frame[task_data.target.name].to_numpy()]

            return dataset_frame

        id_to_dataset_load_function = {
            "-1": lambda: sklearn_load_dataset_function_to_used_data(load_breast_cancer),
            "-2": lambda: sklearn_load_dataset_function_to_used_data(load_digits),
            "-3": lambda: sklearn_load_dataset_function_to_used_data(load_iris),
            "-4": lambda: sklearn_load_dataset_function_to_used_data(load_wine)
        }

        rebuild_benchmark(base_path / "example_benchmark_metatasks",
                          id_to_dataset_load_function=id_to_dataset_load_function)
        rebuild_mts = {}
        for mt_id in mt_ids:
            mt = MetaTask()
            mt.read_metatask_from_files(base_path / "example_benchmark_metatasks", mt_id)
            rebuild_mts[mt_id] = mt

        for mt_id in original_mts.keys():
            org_mt = original_mts[mt_id]
            new_mt = rebuild_mts[mt_id]
            assert org_mt == new_mt

    def test_benchmaker_openml(self):
        base_path = self.base_path
        mt_id = "3"

        omla = OpenMLAssembler(openml_metric_name="area_under_roc_curve", maximize_metric=True, nr_base_models=5)

        # -- Iterate over the task and crawl/build their metatasks
        meta_task = omla.run(mt_id)
        meta_task.to_files(output_dir=base_path / "example_metatasks")

        # -- Build Benchmark
        bmer = BenchMaker(base_path / "example_metatasks", base_path / "example_benchmark_metatasks",
                          manual_filter_duplicates=False, min_number_predictors=5,
                          remove_bad_predictors=True, remove_worse_than_random_predictors=True,
                          remove_constant_predictors=True, tasks_to_use=[mt_id],
                          metric_info=(accuracy_score, "acc", True))

        bmer.build_benchmark(share_data="openml")

        bm_meta_task = MetaTask()
        bm_meta_task.read_metatask_from_files(base_path / "example_benchmark_metatasks", mt_id)

        assert meta_task == bm_meta_task

        rebuild_benchmark(base_path / "example_benchmark_metatasks")

        rebuild_bm_mt = MetaTask()
        rebuild_bm_mt.read_metatask_from_files(base_path / "example_benchmark_metatasks", mt_id)

        assert bm_meta_task == rebuild_bm_mt

    def test_benchmaker_manual_plus_openml_data(self):
        mt_id = "3"

        def get_bm_data(metatask, base_model, preprocessing, inner_split_random_seed):
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
                fold_perfs.append(OpenMLAUROC()(y_test, fold_test_predictions))

            test_confidences, test_predictions, original_indices = zip(
                *sorted(zip(test_confidences, test_predictions, original_indices),
                        key=lambda x: x[2]))
            test_confidences = np.array(test_confidences)
            test_predictions = np.array(test_predictions)

            return test_predictions, test_confidences, all_oof_data, classes_

        # Control Randomness
        random_base_seed_data = 0
        base_rnger = np.random.RandomState(random_base_seed_data)
        random_int_seed_outer_folds = base_rnger.randint(0, 10000000)
        random_int_seed_inner_folds = base_rnger.randint(0, 10000000)

        # Build Metatask and fill dataset
        mt = MetaTask()
        init_dataset_from_task(mt, mt_id)
        mt.read_randomness(random_int_seed_outer_folds, random_int_seed_inner_folds)

        # Folds
        new_fold_indicator = np.zeros(mt.n_instances)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_int_seed_outer_folds)
        for fold_idx, (_, test_indices) in enumerate(
                cv.split(mt.dataset[mt.feature_names], mt.dataset[mt.target_name])):
            new_fold_indicator[test_indices] = fold_idx
        mt.read_folds(new_fold_indicator)

        preproc = get_default_preprocessing()

        base_models = [
            ("RF_4", RandomForestClassifier(n_estimators=4, random_state=0)),
            ("RF_5", RandomForestClassifier(n_estimators=5, random_state=0)),
            ("RF_6", RandomForestClassifier(n_estimators=6, random_state=0)),
            ("RF_7", RandomForestClassifier(n_estimators=7, random_state=0)),
        ]

        for bm_name, bm in base_models:
            bm_predictions, bm_confidences, bm_validation_data, bm_classes = get_bm_data(mt, bm, preproc,
                                                                                         random_int_seed_inner_folds)

            # Sort data
            mt.add_predictor(bm_name, bm_predictions, confidences=bm_confidences,
                             conf_class_labels=list(bm_classes),
                             predictor_description=str(bm), validation_data=bm_validation_data)

        # Save files
        base_path = self.base_path
        mt.to_files(output_dir=base_path / "example_metatasks")

        # -- Build Benchmark
        bmer = BenchMaker(base_path / "example_metatasks", base_path / "example_benchmark_metatasks",
                          manual_filter_duplicates=False, min_number_predictors=2,
                          remove_bad_predictors=True, remove_worse_than_random_predictors=True,
                          remove_constant_predictors=True, tasks_to_use=[mt_id],
                          metric_info=(accuracy_score, "acc", True))

        bmer.build_benchmark(share_data="share_meta_data")

        bm_meta_task = MetaTask()
        bm_meta_task.read_metatask_from_files(base_path / "example_benchmark_metatasks", mt_id)

        assert mt == bm_meta_task

        id_to_dataset_load_function = {
            mt_id: lambda: task_to_dataset(mt_id)
        }

        rebuild_benchmark(base_path / "example_benchmark_metatasks",
                          id_to_dataset_load_function=id_to_dataset_load_function)

        rebuild_bm_mt = MetaTask()
        rebuild_bm_mt.read_metatask_from_files(base_path / "example_benchmark_metatasks", mt_id)

        assert bm_meta_task == rebuild_bm_mt
