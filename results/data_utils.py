import os
import json
import glob
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


def get_id_and_validate_existing_data(base_path):
    # -- Get all existing file paths in metatasks directory
    dir_path_csv = os.path.join(base_path, "metatask_*.csv")
    dir_path_json = os.path.join(base_path, "metatask_*.json")
    data_paths = glob.glob(dir_path_csv)
    meta_data_paths = glob.glob(dir_path_json)

    # -- Merge files for validation
    path_tuples = []
    for csv_path in data_paths:
        csv_name = csv_path[:-4]
        for json_path in meta_data_paths:
            json_name = json_path[:-5]
            if csv_name == json_name:
                path_tuples.append((csv_path, json_path, os.path.basename(csv_name)))
                break

    # - Validate number of pairs
    l_dp = len(data_paths)
    l_mdp = len(meta_data_paths)
    l_pt = len(path_tuples)
    if l_pt < l_mdp or l_pt < l_dp:
        print("Found more files in the preprocessed data directory than file pairs: " +
              "Pairs {}, CSV files: {}, JSON files: {}".format(l_pt, l_dp, l_mdp))

    # - Validate correctness of merge
    for paths_tuple in path_tuples:
        try:
            assert paths_tuple[0][:-4] == paths_tuple[1][:-5]
        except AssertionError:
            raise ValueError("Some data files have wrongly configured names: {} vs. {}".format(paths_tuple[0],
                                                                                               paths_tuple[1]))
    # - Only get task Ids
    return [p_vals[2].rsplit(sep="_", maxsplit=1)[-1] for p_vals in path_tuples]


def get_valid_benchmark_ids(base_path="../results/benchmark_metatasks"):
    file_path_json = os.path.join(base_path, "benchmark_details.json")
    with open(file_path_json) as json_file:
        benchmark_meta_data = json.load(json_file)
    return [int(t_id) for t_id in benchmark_meta_data["valid_task_ids"]]


def get_default_preprocessing():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="constant", fill_value=-1),
             make_column_selector(dtype_exclude="category")),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"),
             make_column_selector(dtype_include="category")),
        ],
        sparse_threshold=0
    )

    return preprocessor
