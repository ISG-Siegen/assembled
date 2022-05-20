"""Example of how to rebuild metatask created using OpenML for a benchmark from a benchmark_details.json

We will store these benchmark metatasks in their own directory.
We re-get the associated metatasks such that a user would not have to do the API calls the related pre-post-processed
metatasks first.
"""

import os
import json

from assembledopenml.openml_assembler import OpenMLAssembler

# -- Read Benchmark Details
file_path_json = os.path.join("../results/openml_benchmark/benchmark_metatasks", "benchmark_details.json")

with open(file_path_json) as json_file:
    benchmark_meta_data = json.load(json_file)

valid_task_ids = benchmark_meta_data["valid_task_ids"]
task_ids_to_valid_predictors = benchmark_meta_data["task_ids_to_valid_predictors"]
selection_constraints_per_task = benchmark_meta_data["selection_constraints_per_task"]

for task_nr, task_id in enumerate(valid_task_ids, start=1):
    print("#### Process Task {} ({}/{}) ####".format(task_id, task_nr, len(valid_task_ids)))
    omla = OpenMLAssembler(openml_metric_name=selection_constraints_per_task[task_id]["openml_metric_name"],
                           maximize_metric=selection_constraints_per_task[task_id]["maximize_metric"],
                           nr_base_models=selection_constraints_per_task[task_id]["nr_base_models"])

    # Crawl Metatask
    valid_predictors = task_ids_to_valid_predictors[str(task_id)]
    mt = omla.rebuild(task_id, valid_predictors)

    # Store to file
    mt.to_files("../results/openml_benchmark/benchmark_output")
    print("Finished Task \n")
