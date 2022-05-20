"""An example on how to build a benchmark from a set of metatasks such that it is filtered, reproducible, and sharable.

This scripts requires that `example_manual_metatasks.py` has been run before.
"""

from assembled.benchmaker import BenchMaker
from assembledopenml.compatibility.openml_metrics import OpenMLAUROC

# -- Create a Benchmark
bmer = BenchMaker(path_to_metatasks="../../results/metatasks",
                  output_path_benchmark_metatask="../../results/example_benchmark/benchmark_metatasks",
                  tasks_to_use=["-1", "-2", "-3", "-4"], min_number_predictors=2,
                  remove_constant_predictors=True, remove_worse_than_random_predictors=True,
                  metric_info=(OpenMLAUROC(), OpenMLAUROC().name, OpenMLAUROC().maximize))
bmer.build_benchmark(share_data="share_prediction_data")


# -- Re-build a Benchmark
# TODO , make it so we can read deatils json and zip and only require dataset as input