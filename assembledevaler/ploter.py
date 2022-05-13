import os
from assembledevaler.plots.stripplots import draw_stripplot
import matplotlib.pyplot as plt
from autorank import plot_stats


class Ploter:
    """Uses data created by Resulter to plot some stuff

    Parameters
    ----------
    path_to_plt_output
        Where to store .pdf versions of plots
    """

    def __init__(self, path_to_plt_output):
        self.path_to_plt_output = path_to_plt_output

    def stripplot(self, fold_performance_data, metric_to_plot, file_postfix="", split=False):
        """

        Parameters
        ----------
        fold_performance_data
            Fold performance data create by Resulter.
        metric_to_plot: str
            Name of the metric that shall be plotted (must be the name of a column in fold_performance_data)
        file_postfix
        split
              If True, split the plot into 2 parts.
        """
        print("\n### Plotting a Stripplot...")
        fold_performance_data = fold_performance_data[
            ["Dataset", "Ensemble Technique", "Fold", metric_to_plot]].set_index("Dataset")

        if not split:
            self._stripplot(fold_performance_data, metric_to_plot, file_postfix)
        else:
            # Split data into 2 plots
            dataset_names = fold_performance_data.index.unique().to_list()
            fold_performance_data_1 = fold_performance_data.loc[dataset_names[:int(len(dataset_names) / 2)]]
            fold_performance_data_2 = fold_performance_data.loc[dataset_names[int(len(dataset_names) / 2):]]
            self._stripplot(fold_performance_data_1, metric_to_plot, file_postfix, "_p1")
            self._stripplot(fold_performance_data_2, metric_to_plot, file_postfix, "_p2")

    def _stripplot(self, data, x_label, file_name_ext, ext=""):
        fig_save_path = os.path.join(self.path_to_plt_output, "results_stripplot{}{}.pdf".format(file_name_ext, ext))

        x_fig = draw_stripplot(data, x=x_label, y=data.index, hue="Ensemble Technique",
                               ylabel="Dataset", xlabel=x_label)
        x_fig.tight_layout()
        plt.savefig(fig_save_path, bbox_inches="tight")
        x_fig.show()

    def autorank_plot(self, autorank_results, ext=""):
        print("\n### Plot Autorank Results")

        fig_save_path = os.path.join(self.path_to_plt_output, "autorank_plot{}.pdf".format(ext))

        fig, ax = plt.subplots(figsize=(12, 8))
        plot_stats(autorank_results, ax=ax)
        plt.title("Autorank Plot for Metric{}".format(ext))
        plt.tight_layout()
        plt.savefig(fig_save_path, bbox_inches="tight")
        plt.show()
