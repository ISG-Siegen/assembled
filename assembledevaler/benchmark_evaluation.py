import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import os

from autorank import autorank, plot_stats, latex_table

from results.data_utils import get_valid_benchmark_ids
from assembledopenml.compatability.openml_metrics import OpenMLAUROC
from assembledevaler.plots.stripplots import draw_stripplot
from assembledevaler.resulter import Resulter


def plot_and_save(i_stripplot_data, i_x_label, save_path):
    x_fig = draw_stripplot(i_stripplot_data, x="Performance", y=i_stripplot_data.index, hue="Ensemble Technique",
                           ylabel="Dataset", xlabel=i_x_label)
    x_fig.tight_layout()
    mlp.pyplot.savefig(save_path)
    x_fig.show()


if __name__ == "__main__":

    # ---- Input parameter
    valid_ids = get_valid_benchmark_ids()  # or [3913, 3560]
    metric = OpenMLAUROC()
    metric_name = metric.name
    keep_sba = True
    keep_vba = True
    normalize_by_closed_vba_sba_gap = True
    gap_bar_plot = False
    split = False
    calc_data_new = True

    # --- Controll
    stripplot_analysis = False
    experimental_analysis = True

    # --- Required data for later
    tmp_file_path = "../results/evaluation/tmp_eval_data.csv"
    file_exists = os.path.isfile(tmp_file_path)
    sba_name = "custom.SingleBest"
    vba_name = "custom.VirtualBest"

    if experimental_analysis:

        # VBA-SBA-Gap Rank Plot

        # Find a good way to aggregate results
        fold_mean_per_dataset = stripplot_data.drop(columns=["Fold"]).reset_index().groupby(
            by=["Dataset", "Ensemble Technique"]).aggregate("mean").reset_index()

        # VBA-SBA-GAP Normalize
        for dataset in fold_mean_per_dataset["Dataset"].unique():
            dataset_subset = fold_mean_per_dataset[fold_mean_per_dataset["Dataset"] == dataset]
            m_vba = dataset_subset[dataset_subset["Ensemble Technique"] == vba_name]["Performance"].iloc[0]
            m_sba = dataset_subset[dataset_subset["Ensemble Technique"] == sba_name]["Performance"].iloc[0]
            fold_mean_per_dataset.loc[fold_mean_per_dataset["Dataset"] == dataset, "Performance"] = dataset_subset[
                "Performance"].apply(lambda x: -(m_sba - x) / (m_sba - m_vba))

        rank_data = fold_mean_per_dataset.pivot(index="Dataset", columns="Ensemble Technique")
        rank_data.columns = rank_data.columns.droplevel(0)
        rank_data = rank_data.drop(columns=[vba_name, sba_name]).reset_index().drop(columns=["Dataset"])

        # To avoid a bug following pivot rename of the column index header
        rank_data = pd.DataFrame(rank_data.values, columns=list(rank_data))

        result = autorank(rank_data, alpha=0.05, verbose=True, order="ascending")  # , order="ascending"
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_stats(result, ax=ax)
        plt.tight_layout()
        plt.show()
        latex_table(result)  # Mean Rank, Median, Mean absolute deviation, Confidence Interval, gamma, Magnitude
        # Gamma is the effect size and magnitude a categorical discretization of it; it shows the difference between
        #   the first place and later placed algorithms.

        # Additional stats
        other_stats = fold_mean_per_dataset.groupby(by=["Ensemble Technique"]).aggregate(["mean", "min", "max", "std",
                                                                                          "var"]).reset_index().sort_values(
            by=("Performance", "mean")).reset_index().drop(columns=["index"])
        print(other_stats.to_string())

        print("\nWins (including ties for the win)")
        win_data = rank_data.rank(axis='columns', ascending=True).stack()
        win_data = win_data[
            win_data.eq(win_data.groupby(level=0).transform('min'))]  # https://stackoverflow.com/a/71041281
        print(win_data.reset_index()["level_1"].value_counts() / len(rank_data))

        # -- now we have a ranking and know who are the best, look at why next

