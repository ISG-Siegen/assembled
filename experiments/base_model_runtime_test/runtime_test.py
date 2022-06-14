"""
Re-run the best model of a task and measure the time it takes to produce the task's results

(requires correct environment, see docker file)
"""

import openml
import time

# Most Costly Task based on number of features and number of instances (and time it took to train ensemble techniques)
openml_task_id = 167124
# 2nd best performing flow (best performing flow can not be reproduced, a build function is missing;
#      is generally buggy we assume it is a bug from development: https://github.com/openml/OpenML/issues/751)
run_id = 10317743
run = openml.runs.get_run(run_id)
flow = openml.flows.get_flow(run.flow_id)
print(flow.dependencies)
print(flow.external_version)


print("Duplicate Model")
model_duplicate = openml.runs.initialize_model_from_run(run_id)
print("Get Task")
task = openml.tasks.get_task(openml_task_id)

print("Run Model on Task")
st = time.time()
run_duplicate = openml.runs.run_model_on_task(model_duplicate, task, avoid_duplicate_runs=False)
print("Time taken:", (time.time() - st) / 60)
print(run_duplicate)
