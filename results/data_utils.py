import os
import json
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


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
