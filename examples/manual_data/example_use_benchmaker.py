"""An example on how to build a benchmark from a set of metatasks such that it is filtered, reproducible, and sharable.

This scripts requires that `example_manual_metatasks.py` has been run before.
"""

from assembled.benchmaker import BenchMaker, rebuild_benchmark
from assembledopenml.compatibility.openml_metrics import OpenMLAUROC

# -- Run to create a Benchmark
bmer = BenchMaker(path_to_metatasks="../../results/metatasks",
                  output_path_benchmark_metatask="../../results/example_benchmark/benchmark_metatasks",
                  tasks_to_use=["-1", "-2", "-3", "-4"], min_number_predictors=2,
                  remove_constant_predictors=True, remove_worse_than_random_predictors=True,
                  metric_info=(OpenMLAUROC(), OpenMLAUROC().name, OpenMLAUROC().maximize))
bmer.build_benchmark(share_data="share_prediction_data")

# # -- Run to re-build a manual Benchmark from just the shareable data and benchmark_details.json
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
import numpy as np


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

rebuild_benchmark("../../results/example_benchmark/benchmark_metatasks",
                  id_to_dataset_load_function=id_to_dataset_load_function)
