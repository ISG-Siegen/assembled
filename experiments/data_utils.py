import os
import json
import glob
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def get_id_and_validate_existing_data():
    # -- Get all existing file paths in metatasks directory
    dir_path_csv = os.path.join("../results/metatasks", "metatask_*.csv")
    dir_path_json = os.path.join("../results/metatasks", "metatask_*.json")
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


def get_valid_benchmark_ids():
    file_path_json = os.path.join("../results/benchmark_metatasks", "benchmark_details.json")
    with open(file_path_json) as json_file:
        benchmark_meta_data = json.load(json_file)
    return [int(t_id) for t_id in benchmark_meta_data["valid_task_ids"]]


def get_benchmark_task_ids_to_dataset_name_and_length():
    valid_task_ids = get_valid_benchmark_ids()
    task_id_to_dataset = {}
    for idx in valid_task_ids:
        file_path_json = os.path.join("../results/benchmark_metatasks", "metatask_{}.json".format(idx))

        task_id_to_dataset[idx] = {}

        with open(file_path_json) as json_file:
            data = json.load(json_file)
            task_id_to_dataset[idx]["dataset_name"] = data["dataset_name"]
            task_id_to_dataset[idx]["n_instances"] = len(data["folds"])
    return task_id_to_dataset


def get_preprocessing_function(cat_feature_names, non_cat_feature_names):
    def preproc_func(X_train, X_test):
        ohe = OneHotEncoder(handle_unknown="ignore", drop="first")

        # no need to fillna for cat features as pandas cat handles it
        X_train_non_cat = X_train[non_cat_feature_names].fillna(-1)
        X_train_cat = pd.DataFrame(ohe.fit_transform(X_train[cat_feature_names]).toarray(),
                                   columns=ohe.get_feature_names_out(), index=X_train.index)
        X_train = pd.concat([X_train_non_cat, X_train_cat], axis=1)

        X_test_non_cat = X_test[non_cat_feature_names].fillna(-1)
        X_test_cat = pd.DataFrame(ohe.transform(X_test[cat_feature_names]).toarray(),
                                  columns=ohe.get_feature_names_out(), index=X_test.index)
        X_test = pd.concat([X_test_non_cat, X_test_cat], axis=1)

        return X_train, X_test

    return preproc_func
