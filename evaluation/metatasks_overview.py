"""Script to generate a table that builds an overview of metatasks.

"""

from tabulate import tabulate
import pandas as pd
from assembledopenml.metatask import MetaTask
from experiments.data_utils import get_valid_benchmark_ids

# Get ids
valid_task_ids = get_valid_benchmark_ids()

# Collect stats for each id
selection_constraints = {}
md_res = []

# Collect data
for task_id in valid_task_ids:
    mt = MetaTask()
    mt.read_metatask_from_files("../results/benchmark_metatasks", task_id)

    # Add metatasks
    nr_class_labels = len(mt.class_labels)
    nr_portfolio_algorithms = len(mt.predictors)
    nr_features = len(mt.feature_names)
    nr_cat_features = len(mt.cat_feature_names)
    nr_instances = len(mt.meta_dataset)
    md_res.append([mt.dataset_name, mt.openml_task_id, nr_instances, nr_portfolio_algorithms, nr_features,
                   nr_cat_features, nr_class_labels])

    # Add selection constraints
    for sel_cons in mt.selection_constraints.keys():
        if sel_cons not in selection_constraints:
            selection_constraints[sel_cons] = set()
        selection_constraints[sel_cons].add(mt.selection_constraints[sel_cons])

# Add total line at the end
# Print Mean Results
res_dict = pd.DataFrame(md_res, columns=["Dataset", "OpenMl Task ID", "#instances", "#portfolio_algorithms", "#feat",
                                         "#cat_feat", "#classes"]).sort_values(by="#instances", ascending=False)
res_dict.loc[len(res_dict)] = ["TOTAL-MEAN", "-"] + [res_dict[col].mean() for col in list(res_dict)[2:]]
print(tabulate(res_dict, headers="keys", tablefmt="psql", showindex=False))
print("\n The following selection constrains have been found: {}".format(selection_constraints))
res_dict.to_csv("../results/evaluation/benchmark_metatask_overview.csv", index=False)

# Re-order
res_dict = res_dict[["Dataset", "OpenMl Task ID", "#instances", "#feat", "#classes", "#portfolio_algorithms"]]
res_dict.to_latex("../results/evaluation/benchmark_metatask_overview.tex", index=False)
