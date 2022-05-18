""" Example for OpenML-CC18

The following examples show how to use the code of Assembled-OpenML for a set of OpenML task IDs based on a tag
"""

import openml
from assembledopenml.openml_assembler import OpenMLAssembler

# -- Select a subst of tasks id for which you want to build metatasks
# - use the set of tasks from OpenML-CC18, see https://docs.openml.org/benchmark/#openml-cc18
tasks = openml.tasks.list_tasks(tag="OpenML-CC18", output_format="dataframe")
task_list = tasks["tid"].tolist()  # or just use: [3, 6, 11, 12, 14, 15]

# -- Init Assembler
# The settings of the OpenMLAssembler make us look for the top 50 base models based on the metric area_under_roc_curve
omla = OpenMLAssembler(openml_metric_name="area_under_roc_curve", maximize_metric=True, nr_base_models=50)

# -- Iterate over the task and crawl/build their metatasks

for t_idx, task_id in enumerate(task_list, 1):
    # Build meta-dataset for each task
    print("####### Process Task {} ({}/{}) #######".format(task_id, t_idx, len(task_list)))
    meta_task = omla.run(task_id)
    meta_task.to_files(output_dir="../results/metatasks")
