import os
import pandas as pd
import numpy as np


def save_fold_results(y_true, y_pred, fold_idx, out_path, technique_name, index_metatask, classification=True):
    """Code to Save Fold results in a file such that we can add any other folds results in the future

    Requires sequential evaluation/saving. Hence, can not be used in parallel environments.
    """

    if out_path is None:
        return

    # Path test
    path_exists = os.path.exists(out_path)

    # Get data that is to be saved
    fold_indicator = [fold_idx for _ in range(len(y_true))]
    res_df = pd.DataFrame(np.array([y_true, index_metatask, fold_indicator, y_pred]).T,
                          columns=["ground_truth", "Index-Metatask", "Fold", technique_name])

    # Keep type correct
    if classification:
        # Note: while we tell read_csv that technique name should be read as string, it does not matter if
        #   technique name is not in the file to be read. Hence it works for loading it initially
        input_dtype = "string"
        res_df = res_df.astype(input_dtype).astype({"Index-Metatask": int, "Fold": int})
    else:
        input_dtype = None

    # Save folds base
    if fold_idx == 0:
        if path_exists:
            # Load old data
            tmp_df = pd.read_csv(out_path, dtype=input_dtype).astype(
                {"Index-Metatask": int, "Fold": int}).set_index(["ground_truth", "Index-Metatask", "Fold"])
            # Pre-fill technique values
            tmp_df[technique_name] = pd.Series(np.nan, index=np.arange(len(tmp_df)))
            # Insert new values into data
            res_df = res_df.set_index(["ground_truth", "Index-Metatask", "Fold"])

            # Try-catch to detect (potential) random seed error
            try:
                tmp_df.loc[res_df.index, technique_name] = res_df
            except KeyError:
                raise KeyError("Wrong fold/index/ground-truth combinations found in results file." +
                               " Did you create the existing results file with a different random seed for" +
                               " the meta train test split? " +
                               " To fix this delete the existing files or adjust the random seed.")

            tmp_df.reset_index().sort_values(by=["Index-Metatask"]).to_csv(out_path, index=False)
        else:
            res_df.to_csv(out_path, index=False)

    else:
        # Check error
        if not path_exists:
            raise RuntimeError("Somehow a later fold is trying to be saved first. " +
                               "Did the output file got deleted between folds?")

        # Read so-far data
        tmp_df = pd.read_csv(out_path, dtype=input_dtype).astype(
            {"Index-Metatask": int, "Fold": int}).set_index(["ground_truth", "Index-Metatask", "Fold"])

        if len(list(tmp_df)) == 1:
            # Initial case, first time file building
            res_df.to_csv(out_path, mode='a', header=False, index=False)
        else:
            # Default case later, same as above wihtout init
            res_df = res_df.set_index(["ground_truth", "Index-Metatask", "Fold"])
            tmp_df.loc[res_df.index, technique_name] = res_df
            tmp_df.reset_index().sort_values(by=["Index-Metatask"]).to_csv(out_path, index=False)
