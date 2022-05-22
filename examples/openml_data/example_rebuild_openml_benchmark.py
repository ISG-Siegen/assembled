"""Example of how to rebuild metatask created using OpenML for a benchmark from a benchmark_details.json
"""
from assembled.benchmaker import rebuild_benchmark

# -- Read Benchmark Details
benchmark_data_dir = "../../results/openml_benchmark/benchmark_metatasks"
# Stores rebuild metatask in benchmark_data_dir directory
#   (code reads from the benchmark details that it is a full openml metatask dataset)
rebuild_benchmark(benchmark_data_dir)
