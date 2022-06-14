"""Example of how to rebuild metatask for a benchmark for a benchmark_details.json

The following examples show how to re-build a benchmark using the benchmark metadata provided by us.

The metatasks that this script creates, contain the results of post-processing all meta-tasks crawled using
the tag "OpenML-CC18".

We will store these benchmark metatasks in their own directory.
We re-crawl the associated metatasks such that a user would not have to crawl the related pre-post-processed
metatasks first.
"""

import os
import json

from assembledopenml.openml_crawler import OpenMLCrawler

# -- Read Benchmark Details
file_path_json = os.path.join("../results/benchmark_metatasks", "benchmark_details.json")

with open(file_path_json) as json_file:
    benchmark_meta_data = json.load(json_file)
valid_task_ids = benchmark_meta_data["valid_task_ids"]
task_ids_to_valid_predictors = benchmark_meta_data["task_ids_to_valid_predictors"]
selection_constraints = benchmark_meta_data["selection_constraints"]

# -- Crawl Original Metatask and remove non-valid predictors
omlc = OpenMLCrawler(openml_metric_name=selection_constraints["openml_metric_name"],
                     maximize_metric=selection_constraints["maximize_metric"],
                     nr_base_models=selection_constraints["nr_base_models"])

nr_tasks = len(valid_task_ids)
for task_nr, task_id in enumerate(valid_task_ids, start=1):
    print("#### Process Task {} ({}/{}) ####".format(task_id, task_nr, nr_tasks))

    # Crawl Metatask
    valid_predictors = task_ids_to_valid_predictors[str(task_id)]
    mt = omlc.rebuild(task_id, valid_predictors)

    # Store to file
    mt.to_files("../results/benchmark_metatasks")
    print("Finished Task \n")
