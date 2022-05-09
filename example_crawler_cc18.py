""" Cralwer Example for OpenML-CC18

The following examples show how to use the crawler of Assembled-OpenML for a set of OpenML task IDs based on a tag
"""

import openml
from assembledopenml.openml_crawler import OpenMLCrawler

# -- Select a subst of tasks id for which you want to build metatasks
# - use the set of tasks from OpenML-CC18, see https://docs.openml.org/benchmark/#openml-cc18
tasks = openml.tasks.list_tasks(tag="OpenML-CC18", output_format="dataframe")
task_list = tasks["tid"].tolist()

# -- Init Crawler
omlc = OpenMLCrawler(openml_metric_name="area_under_roc_curve", maximize_metric=True, min_runs=10,
                     test_run=True)

# -- Iterate over the task and crawl/build their metatasks

for t_idx, task_id in enumerate(task_list, 1):
    # Build meta-dataset for each task
    print("####### Process Task {} ({}/{}) #######".format(task_id, t_idx, len(task_list)))

    # Filter the Task on OpemML based on area_under_roc_curve performance (which needs to be maximized)
    # Moreover, all selected flows/example_algorithms should have at least 10 different runs/configurations.
    # Lastly, only do a test run, that is, limit the request size to a small subset of all possible results.

    meta_task = omlc.run(task_id)
    meta_task.to_files(output_dir="./metatasks")
