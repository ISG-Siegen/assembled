import os
from pathlib import Path
import pandas as pd
import numpy as np
import glob
from shutil import rmtree
import json


# -- Code for parallel usage
def _get_savable_df(y_col_name, y_to_save, index_metatask, fold_idx, classification):
    fold_indicator = [fold_idx for _ in range(len(y_to_save))]
    res_df = pd.DataFrame(np.array([y_to_save, index_metatask, fold_indicator]).T,
                          columns=[y_col_name, "Index-Metatask", "Fold"])

    # Keep type correct
    if classification:
        input_dtype = "string"
    else:
        input_dtype = "float64"

    return res_df.astype({y_col_name: input_dtype, "Index-Metatask": int, "Fold": int})


def _merge_saved_files(y_files, classification):
    if classification:
        input_dtype = "string"
    else:
        input_dtype = "float64"

    tmp_df = None

    for fold_file in y_files:
        if tmp_df is None:
            tmp_df = pd.read_csv(fold_file, dtype=input_dtype).astype({"Index-Metatask": int, "Fold": int})
        else:
            tmp_df = pd.concat(
                [tmp_df, pd.read_csv(fold_file, dtype=input_dtype).astype({"Index-Metatask": int, "Fold": int})],
                axis=0)

    return tmp_df


def _save_fold_results(y_true, y_pred, fold_idx, out_path, technique_name, index_metatask, task_id,
                       classification=True):
    base_path = Path(out_path).joinpath("results_for_metatask_{}".format(task_id))
    base_path.mkdir(exist_ok=True)

    # -- Check if y_true is already saved
    path_to_evaluation_y_true = base_path.joinpath("F{}_evaluation_y_true.csv".format(fold_idx))
    if not path_to_evaluation_y_true.exists():
        # Save fold y_true
        _get_savable_df("ground_truth", y_true, index_metatask,
                        fold_idx, classification).to_csv(path_to_evaluation_y_true, index=False)

    # -- Create directory for result of this technique (if needed)
    fold_results_dir = base_path.joinpath(technique_name)
    fold_results_dir.mkdir(exist_ok=True)

    # -- Save technique y_pred
    path_to_fold_y_pred = fold_results_dir.joinpath("F{}_{}.csv".format(fold_idx, technique_name))
    _get_savable_df(technique_name, y_pred, index_metatask, fold_idx, classification).to_csv(path_to_fold_y_pred,
                                                                                             index=False)


def _save_metadata(output_dir_path, task_id, technique_name, run_meta_data, fold_idx):
    base_path = Path(output_dir_path).joinpath("results_for_metatask_{}".format(task_id)).joinpath(technique_name)
    fold_metadata_path = base_path.joinpath("F{}_metadata_{}.json".format(fold_idx, technique_name))

    with open(fold_metadata_path, "w") as f:
        json.dump({str(fold_idx): run_meta_data, "name": technique_name}, f)


def _collect_metadata_from_files(metadata_files):
    # Load all parallel stored metadata
    collected_metadata = {}
    for metadata_f in metadata_files:
        with open(metadata_f, "r") as f:
            collected_metadata = {**collected_metadata, **json.load(f)}

    # Get sequential format from files
    t_name = collected_metadata.pop("name")
    folds_sorted = sorted([int(i) for i in collected_metadata.keys()])
    metadata_keys = list(collected_metadata[str(folds_sorted[0])].keys())

    return {t_name: {k: [collected_metadata[str(i)][k] for i in folds_sorted] for k in metadata_keys}}


def merge_fold_results(out_path, task_id, classification=True, clean_up=True):
    """For this to work, we assume that y_true for each fold was successfully saved.

    Metadata is merged too, if metadata files exist in the task directory.
    """

    base_path = Path(out_path).joinpath("results_for_metatask_{}".format(task_id))

    if not base_path.exists():
        raise ValueError("No data directory to merge the result for metatask {} exist.".format(task_id))

    # --- Get and merge y_true
    y_true_files = glob.glob(str(base_path.joinpath("F*_evaluation_y_true.csv")))
    results_df = _merge_saved_files(y_true_files, classification)
    sanity_length = len(results_df)

    # --- Get base model results
    base_model_result_dirs = glob.glob(os.path.join(base_path, "*", ""))
    all_metadata = {}
    for bm_res_dir in base_model_result_dirs:
        bm_y_pred = _merge_saved_files(glob.glob(bm_res_dir + "/*.csv"), classification)
        results_df = results_df.merge(bm_y_pred, on=["Index-Metatask", "Fold"], validate="one_to_one", how="outer")
        all_metadata = {**all_metadata, **_collect_metadata_from_files(glob.glob(bm_res_dir + "/*.json"))}

    # --- Save merged results
    if len(results_df) != sanity_length:
        raise ValueError("Something went wrong during merges.")

    results_df.sort_values(by=["Index-Metatask"]).to_csv(Path(out_path).joinpath(
        "results_for_metatask_{}.csv".format(task_id)), index=False)

    with open(Path(out_path).joinpath("evaluation_metadata_for_metatask_{}.json".format(task_id)), "w") as f:
        json.dump(all_metadata, f)

    if clean_up:
        rmtree(base_path)


# -- Sequential Usage Special Case
def _save_fold_results_sequentially(y_true, y_pred, fold_idx, out_path, technique_name, index_metatask, task_id,
                                    classification=True):
    """Code to Save Fold results in a file such that we can add any other folds results in the future

    Requires sequential evaluation/saving. Hence, can not be used in parallel environments.
    """

    # Path test
    base_path = Path(out_path).joinpath("results_for_metatask_{}.csv".format(task_id))
    path_exists = base_path.exists()

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
            tmp_df = pd.read_csv(base_path, dtype=input_dtype).astype(
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

            tmp_df.reset_index().sort_values(by=["Index-Metatask"]).to_csv(base_path, index=False)
        else:
            res_df.to_csv(base_path, index=False)

    else:
        # Check error
        if not path_exists:
            raise RuntimeError("Somehow a later fold is trying to be saved first. " +
                               "Did the output file got deleted between folds?")

        # Read so-far data
        tmp_df = pd.read_csv(base_path, dtype=input_dtype).astype(
            {"Index-Metatask": int, "Fold": int}).set_index(["ground_truth", "Index-Metatask", "Fold"])

        if len(list(tmp_df)) == 1:
            # Initial case, first time file building
            res_df.to_csv(base_path, mode='a', header=False, index=False)
        else:
            # Default case later, same as above wihtout init
            res_df = res_df.set_index(["ground_truth", "Index-Metatask", "Fold"])
            tmp_df.loc[res_df.index, technique_name] = res_df
            tmp_df.reset_index().sort_values(by=["Index-Metatask"]).to_csv(base_path, index=False)


def _save_metadata_sequentially(output_dir_path, task_id, technique_name, fold_metadata, fold_idx):
    base_path = Path(output_dir_path).joinpath("evaluation_metadata_for_metatask_{}.json".format(task_id))
    path_exists = base_path.exists()

    # Get data so far
    if path_exists:
        with open(base_path, "r") as f:
            eval_md = json.load(f)
    else:
        eval_md = {}

    # Add new Data
    if technique_name not in eval_md:
        eval_md[technique_name] = {}

    for metadata_key in fold_metadata:
        if metadata_key not in eval_md[technique_name]:
            eval_md[technique_name][metadata_key] = []

        if len(eval_md[technique_name][metadata_key]) > fold_idx:
            raise ValueError("Adding New Fold Metadata to existing metadata with more entries than the current fold "
                             + "number. Found {} many entires. Wanted to add metadata for fold: {}. ".format(
                len(eval_md[technique_name][metadata_key]), fold_idx)
                             + "This happened most likely because old metadata files were not deleted!")

        eval_md[technique_name][metadata_key].append(fold_metadata[metadata_key])

    # Write data
    with open(base_path, "w") as f:
        json.dump(eval_md, f)


def save_results(output_dir_path, store_results, save_evaluation_metadata, ensemble_test_y, y_pred_ensemble_model,
                 fold_idx, technique_name, test_indices, task_id, run_meta_data):
    if output_dir_path is not None:
        if store_results == "sequential":
            _save_fold_results_sequentially(ensemble_test_y, y_pred_ensemble_model, fold_idx,
                                            output_dir_path, technique_name, test_indices,
                                            task_id)
        else:
            _save_fold_results(ensemble_test_y, y_pred_ensemble_model, fold_idx, output_dir_path,
                               technique_name, test_indices, task_id)

        if save_evaluation_metadata:
            if store_results == "sequential":
                _save_metadata_sequentially(output_dir_path, task_id, technique_name,
                                            run_meta_data, fold_idx)
            else:
                _save_metadata(output_dir_path, task_id, technique_name, run_meta_data, fold_idx)
