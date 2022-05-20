""" A script to check the bad predictors of crawled metatasks.

We have used this to find the KNOWN_BAD_FLOWS below for a subset of all crawled metatasks.
This might be helpful to explore the problems of OpenML prediction data in the future.
"""

from assembled.benchmaker import get_id_and_validate_existing_data
from assembled.metatask import MetaTask
from assembledopenml.metaflow import MetaFlow
import pandas as pd
from tabulate import tabulate

# Known Bad flows for which we do not know the reasons nor a workaround to "fix" the bad predictions
KNOWN_BAD_FLOWS = [19030, 19037, 19039, 19035, 18818, 17839, 17761]


def get_bad_rows_df(meta_flow, mf_not_equal_at_all_idx):
    mf_not_equal_at_all_idx = [x for x, y in mf_not_equal_at_all_idx]
    pred_wrong = meta_flow.predictions.loc[mf_not_equal_at_all_idx]
    confs_wrong = meta_flow.confidences.loc[mf_not_equal_at_all_idx]

    return pd.concat([pred_wrong, confs_wrong], axis=1)


# Get ids
base_path = "../../results/metatasks"
valid_task_ids = get_id_and_validate_existing_data(base_path)  # or set valid ids directly: [3, 6, 11, 12, 14, 15]

# Collect stats for each id
selection_constraints = {}
md_res = []

# Collect data
for task_id in valid_task_ids:
    mt = MetaTask()
    mt.read_metatask_from_files(base_path, task_id)

    if not mt.bad_predictors:
        continue

    print(task_id, "| Nr. of bad predictors", len(mt.bad_predictors), "| Nr total predictors", len(mt.predictors))

    bad_with_reason = {}
    # Validate bad predictors
    for bad_pred in mt.bad_predictors:
        print("Name:", bad_pred)
        print("Description:", mt.predictor_descriptions[bad_pred])
        print("Corruption Details:", mt.predictor_corruptions_details[bad_pred])
        # Read Flow and Run ID
        n_split = bad_pred.split("_")
        flow_id = int(n_split[2])
        run_id = int(n_split[-1])

        if flow_id in KNOWN_BAD_FLOWS:
            print("Flow ID is known to behave badly. The reason is unknown.")
            continue

        # Re-Build metaflow
        mf = MetaFlow(flow_id, mt.predictor_descriptions[bad_pred], 0, run_id)
        mf.get_predictions_data(mt.class_labels)
        print("File URL:", mf.predictions_url)

        # Find bad rows
        conf_preds = mf._confidence_to_predictions()
        tol_equal, not_equal_at_all_idx, equal_with_tol_idx = mf._gather_wrong_confidences(conf_preds)

        not_equal_at_all = get_bad_rows_df(mf, not_equal_at_all_idx)
        equal_with_tol = get_bad_rows_df(mf, equal_with_tol_idx)

        if not not_equal_at_all.empty:
            print("\n\n ### NOT EQUAL AT ALL ROWS RANDOM EXAMPLE ###")
            n = min(5, len(not_equal_at_all))
            print(tabulate(not_equal_at_all.sample(n=n), floatfmt=".4f", headers='keys', tablefmt='psql'))

        answer_user = input("Do you see a reason for wrong predictions? (y/n) \n")
        if answer_user == "y":
            reason = input("Reason?")
            bad_with_reason[bad_pred] = reason
        elif answer_user in ["n", ""]:
            # Stay as bad predictor
            pass
        else:
            raise ValueError("Unable to understand your answer.")

    print(task_id, bad_with_reason)
    # TODO implement a tool that filters/updates bad predictors (maybe)
