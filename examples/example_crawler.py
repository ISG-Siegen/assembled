"""Crawler Example

The following examples show how to use the crawler of Assembled-OpenML for specified set of OpenML task IDs
"""

from assembledopenml.openml_crawler import OpenMLCrawler

# -- Select a subst of tasks id for which you want to build metatasks
task_list = [3, 6, 11, 12, 14, 15]

# -- Init Crawler
omlc = OpenMLCrawler(nr_base_models=10)

# -- Iterate over the task and crawl/build their metatasks
for t_idx, task_id in enumerate(task_list, 1):
    # Build meta-dataset for each task
    print("####### Process Task {} ({}/{}) #######".format(task_id, t_idx, len(task_list)))
    meta_task = omlc.run(task_id)
    meta_task.to_files(output_dir="../results/metatasks")
