import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt

from experiments.data_utils import get_valid_benchmark_ids, get_benchmark_task_ids_to_dataset_name_and_length
from assembledopenml.compatability.openml_metrics import no_conf_openml_area_under_roc_curve as auroc
from evaluation.evaluation_helpers import draw_stripplot


def plot_and_save(i_stripplot_data, i_x_label, save_path):
    x_fig = draw_stripplot(i_stripplot_data, x="Performance", y=i_stripplot_data.index, hue="Ensemble Technique",
                           ylabel="Dataset", xlabel=i_x_label)
    x_fig.tight_layout()
    mlp.pyplot.savefig(save_path)
    x_fig.show()


if __name__ == "__main__":

    # ---- Input parameter
    valid_ids = get_valid_benchmark_ids()  # or [3913, 3560]
    metric = auroc
    metric_name = "AUROC"  # or use auroc.__name__ (but that might be bad depending on the used metric)
    keep_sba = True
    keep_vba = True
    normalize_by_closed_vba_sba_gap = False
    gap_bar_plot = False
    split = False

    # --- Required data for later
    simple_eval_data = pd.DataFrame
    stripplot_data_cols = ["Dataset", "Fold", "Performance", "Ensemble Technique", "n_instances"]
    stripplot_data = pd.DataFrame(columns=stripplot_data_cols)
    task_id_to_dataset = get_benchmark_task_ids_to_dataset_name_and_length()

    # ---- collect data
    for idx in valid_ids:
        ensemble_prediction_data = pd.read_csv("../results/benchmark_output/results_for_metatask_{}.csv".format(idx))
        dataset_name = task_id_to_dataset[idx]["dataset_name"]
        n_instances = task_id_to_dataset[idx]["n_instances"]
        fold_perf = []

        for fold_i in ensemble_prediction_data["Fold"].unique().tolist():
            fold_data = ensemble_prediction_data[ensemble_prediction_data["Fold"] == fold_i]
            fold_y_true = fold_data["ground_truth"]
            pred_data = fold_data.drop(columns=["ground_truth", "Index-Metatask", "Fold"])

            fold_performances = pred_data.apply(lambda x: metric(fold_y_true, x), axis=0)

            if normalize_by_closed_vba_sba_gap:
                # Normalized by vba-sba-gap
                m_vba = fold_performances["DCS_VBA"]
                m_sba = fold_performances["DCS_SBA"]
                fold_performances = fold_performances.apply(lambda x: (m_sba - x) / (m_sba - m_vba))

            for ensemble_technique_name, f_p in zip(fold_performances.index.tolist(), fold_performances.tolist()):
                stripplot_data = pd.concat([stripplot_data, pd.DataFrame([[dataset_name, fold_i, f_p,
                                                                           ensemble_technique_name, n_instances]],
                                                                         columns=stripplot_data_cols)],
                                           axis=0)

    # -- Post process
    stripplot_data = stripplot_data.sort_values(by=["n_instances", "Ensemble Technique"], ascending=False).drop(
        columns=["n_instances"]).set_index("Dataset")

    if not keep_sba:
        stripplot_data = stripplot_data[stripplot_data["Ensemble Technique"] != "DCS_SBA"]
    if not keep_vba:
        stripplot_data = stripplot_data[stripplot_data["Ensemble Technique"] != "DCS_VBA"]

    x_label = metric_name
    if normalize_by_closed_vba_sba_gap:
        x_label = "Closed VBA-SBA-GAP for " + x_label

    fig_name_ext = "_vba_sba_gap" if normalize_by_closed_vba_sba_gap else ""
    fig_save_path = "../results/evaluation/results_stripplot{}.pdf".format(fig_name_ext)

    if not split:
        plot_and_save(stripplot_data, x_label, fig_save_path)
    else:
        dataset_names = stripplot_data.index.unique().to_list()
        part_1 = dataset_names[:int(len(dataset_names) / 2)]
        part_2 = dataset_names[int(len(dataset_names) / 2):]
        stripplot_data_1 = stripplot_data.loc[part_1]
        stripplot_data_2 = stripplot_data.loc[part_2]
        plot_and_save(stripplot_data_1, x_label,
                      "../results/evaluation/results_stripplot{}_p1.pdf".format(fig_name_ext))
        plot_and_save(stripplot_data_2, x_label,
                      "../results/evaluation/results_stripplot{}_p2.pdf".format(fig_name_ext))

    # -- Additional plots
    if normalize_by_closed_vba_sba_gap and gap_bar_plot:
        # Group stripplot data
        bar_plot_data = stripplot_data.drop(columns=["Fold"]).reset_index()
        bar_plot_data = bar_plot_data.groupby(by=["Dataset", "Ensemble Technique"]).aggregate("mean").reset_index()
        bar_plot_data = bar_plot_data.pivot(index="Dataset", columns="Ensemble Technique")
        bar_plot_data.columns = bar_plot_data.columns.droplevel(0)

        ax = bar_plot_data.plot.barh(figsize=(20, 10))
        ax.set_ylabel("Dataset")
        ax.set_xlabel(x_label)
        plt.tight_layout()
        plt.show()
