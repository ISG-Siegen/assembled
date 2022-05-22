import pathlib
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine

from assembled.metatask import MetaTask
from assembled.benchmaker import BenchMaker, rebuild_benchmark
from assembledopenml.openml_assembler import OpenMLAssembler


class TestBenchMaker:

    def test_benchmaker(self):
        base_path = pathlib.Path(__file__).parent.resolve()
        mt_ids = ["-1", "-2", "-3", "-4"]

        # -- Build Benchmark
        bmer = BenchMaker(base_path / "example_metatasks", base_path / "example_benchmark_metatasks",
                          manual_filter_duplicates=False, min_number_predictors=5,
                          remove_bad_predictors=True, remove_worse_than_random_predictors=True,
                          remove_constant_predictors=True, tasks_to_use=mt_ids,
                          metric_info=(accuracy_score, "acc", True))

        bmer.build_benchmark(share_data="share_prediction_data")

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
        base_path = pathlib.Path(__file__).parent.resolve()
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
        # TODO: add code to test if i can rebuild a metatask with openml data from openml task id

        print()
