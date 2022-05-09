import pandas as pd
import os
import json
import glob
import hashlib
import pathlib
from typing import Optional


class Resulter:
    """Base Class to Handle, Read, and Format Output of Benchmark

    Parameters
    ----------
    metatask_ids
    path_to_metatask
        Used to read the size of the metatasks (and other metadata if needed)
    path_to_benchmark_output
        Used to read the predictions of different ensembles
    """

    def __init__(self, metatask_ids, path_to_metatask, path_to_benchmark_output, classification=True):
        self.metatask_ids = metatask_ids
        self.path_to_metatask = path_to_metatask
        self.path_to_benchmark_output = path_to_benchmark_output
        self.metatask_metadata = self.get_metatask_metadata(metatask_ids, path_to_metatask)
        self.classification = classification
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")

        if not self.classification:
            raise NotImplementedError("Regression not supported yet.")

    @staticmethod
    def get_metatask_metadata(metatask_ids, path_to_metatask):
        mtid_to_metadata = {}
        for mt_id in metatask_ids:
            file_path_json = os.path.join(path_to_metatask, "metatask_{}.json".format(mt_id))
            mtid_to_metadata[mt_id] = {}

            # Collect metadata relevant for us
            with open(file_path_json) as json_file:
                data = json.load(json_file)
                mtid_to_metadata[mt_id]["dataset_name"] = data["dataset_name"]
                mtid_to_metadata[mt_id]["n_instances"] = len(data["folds"])

        return mtid_to_metadata

    def read_benchmark_output_for_metatask(self, mt_id):
        input_path = os.path.join(self.path_to_benchmark_output, "results_for_metatask_{}.csv".format(mt_id))

        if self.classification:
            df = pd.read_csv(input_path, dtype="string").astype({"Index-Metatask": int, "Fold": int})
        else:
            raise NotImplementedError

        return df

    def get_fold_performance_data(self, metric, metric_name: Optional[str] = None, use_cache=True):
        """Build a dataframe containing the metric score for each fold for all ensemble techniques for all metatasks.

        Parameters
        ----------
        metric
            sklearn-like metric: takes as input (y_true, y_pred)
        metric_name: str, default=None
            If None, we will call metric.name to get the name of the metric.
        use_cache: bool, default=True
            If True, we test if we already have the desired dataframe in the cache and that results did not change.
            Moreover, if it does not exist in the cache and use_cache=True, the results are cached.

        Returns
        -------
        fold_performance_data: DataFrame, columns=["Fold", "Ensemble Technique", metric_name, "Dataset"]
            By default, the returned dataframe is sorted by n_instances descending.
        """
        # -- Check if the data is already in teh cache
        if use_cache:
            cache_res = self.check_cache("fold_performance_data")
            if cache_res is not None:
                return cache_res
        print("\n### Build data from Benchmark Output")

        # -- Otherwise compute the value
        metric_name = metric.name if metric_name is None else metric_name
        re_df = pd.DataFrame(columns=["Fold", "Ensemble Technique", metric_name, "Dataset", "n_instances"])

        def fold_data_to_performance(fold_data):
            y_true = fold_data["ground_truth"]
            return fold_data.drop(columns=["Index-Metatask", "Fold", "ground_truth"]).apply(lambda y_pred:
                                                                                            metric(y_true, y_pred))

        for mt_id in self.metatask_ids:
            mt_res = self.read_benchmark_output_for_metatask(mt_id)

            # Get Performance (cols: Fold,technique 1, technique 2, ...)
            mt_res = mt_res.groupby("Fold").apply(fold_data_to_performance).reset_index()

            # Transformed into transaction like format: (cols: Fold, Ensemble Technique Name, metric_name)
            mt_res = mt_res.melt(id_vars=["Fold"], var_name="Ensemble Technique", value_name=metric_name)

            # Add required dataset metadata
            mt_res["Dataset"] = self.metatask_metadata[mt_id]["dataset_name"]
            mt_res["n_instances"] = self.metatask_metadata[mt_id]["n_instances"]

            # Add to overall data
            re_df = pd.concat([re_df, mt_res], axis=0)

        # -- Post-processing
        # Sort by n_instances for nicer plots
        re_df = re_df.sort_values(by=["n_instances", "Ensemble Technique"], ascending=False).drop(
            columns=["n_instances"])

        # Handle dtypes and such
        re_df = re_df.astype({"Fold": "int64", metric_name: "float64"}).reset_index(drop=True)

        if use_cache:
            self.save_to_cache(re_df, "fold_performance_data")

        return re_df

    @staticmethod  # TODO might add evaler as input such that these input values can be reduced
    def get_vb_sb_normalization(fp_df, metric_name, vb_name, sb_name):
        # WARNING:
        #   We do not compute the gap based on the fold-dataset group but on the whole dataset group
        #   We do this because we can only guarantee that a gap exist for the whole dataset (per our benchmark
        #       preprocessing). If no gap exists, the metric breaks and can not be correctly aggregated.
        #   Depending on the fold, it can quickly happen that no gap exists.

        re_df = fp_df.copy()

        # Normalization code
        def norm_vb_sb_gap(group):
            m_sb = group.loc[group["Ensemble Technique"] == sb_name, "MPF"].iloc[0]
            m_vb = group.loc[group["Ensemble Technique"] == vb_name, "MPF"].iloc[0]

            group["VB_SB_GAP"] = group["MPF"].apply(
                lambda x: (m_sb - x) / (m_sb - m_vb))

            return group

        # Get Means but "spread across all fold"
        fold_means = re_df.drop(columns=["Fold"]).groupby(by=["Dataset", "Ensemble Technique"]).aggregate(
            "mean").reset_index().set_index(["Dataset", "Ensemble Technique"])[metric_name]
        re_df = re_df.set_index(["Dataset", "Ensemble Technique"])
        re_df["MPF"] = fold_means
        re_df = re_df.reset_index()

        return re_df.groupby(["Fold", "Dataset"]).apply(norm_vb_sb_gap).drop(columns=["MPF"])

    @staticmethod
    def get_relative_improvement_to_sb(fp_df, metric_name, sb_name):
        re_df = fp_df.copy()

        def ri_sb(fold_dataset_group):
            m_sb = fold_dataset_group.loc[fold_dataset_group["Ensemble Technique"] == sb_name, metric_name].iloc[0]

            def ri(x):
                return x / m_sb - 1

            fold_dataset_group["RI_SB"] = fold_dataset_group["AUROC"].apply(ri)
            return fold_dataset_group

        return re_df.groupby(["Fold", "Dataset"]).apply(ri_sb)

    # --- Post Processing
    def filter_fp_data(self, fold_performance_data, reduce_to_ids):
        valid_dataset_names = [self.metatask_metadata[mt_id]["dataset_name"] for mt_id in reduce_to_ids]

        return fold_performance_data[fold_performance_data["Dataset"].isin(valid_dataset_names)]

    # --- cache stuff
    @property
    def results_checksum(self):
        file_names = [os.path.join(self.path_to_benchmark_output, "results_for_metatask_{}.csv".format(mt_id))
                      for mt_id in self.metatask_ids]
        combined_hashs = str([hashlib.md5(open(f_name, 'rb').read()).digest() for f_name in file_names]).encode("utf-8")
        return hashlib.md5(combined_hashs).hexdigest()

    def get_cache_save_path(self, df_name):
        save_name = "{}_{}.csv".format(df_name, self.results_checksum)
        return os.path.join(self.cache_dir, save_name)

    def check_cache(self, df_name):

        # Return DF if file exists
        path_to_file = self.get_cache_save_path(df_name)
        if os.path.exists(path_to_file):
            print("\n### Found Correct Data in Cache {}, Loading it...".format(path_to_file))
            return pd.read_csv(path_to_file)

    def save_to_cache(self, df, df_name):
        # Clean other files of this type
        for existing_path in glob.glob(os.path.join(self.cache_dir, "{}_*.csv".format(df_name))):
            os.remove(existing_path)

        # Save
        df.to_csv(self.get_cache_save_path(df_name), index=False)
