import pandas as pd
from tabulate import tabulate
from autorank import autorank, latex_table


class Evaler:
    """Evaler enables easy access to different evaluations methods

    Parameters
    ----------
    vb_name
    sb_name
    other_oracles
    """

    def __init__(self, vb_name="custom.VirtualBest", sb_name="custom.SingleBest",
                 other_oracles=()):
        self.vb_name = vb_name
        self.sb_name = sb_name
        self.other_oracles = other_oracles

        self.n_rounds = 3

    # -- Finding the Average Best Method using Ranking and Aggregation of Normalized Scores
    @staticmethod
    def transpose_means(data):
        re_data = data.pivot(index="Dataset", columns="Ensemble Technique")
        re_data.columns = re_data.columns.droplevel(0)
        return re_data

    def use_autorank(self, fold_performance_data, metric_name, metric_maximize, approach="frequentist", keep_vb=False):
        print("\n### Use Autorank to produce evaluations for metric {} and approach {}...".format(metric_name,
                                                                                                  approach))

        # --- Transform fold_performance_data into data for autorank
        rank_data = fold_performance_data_to_fold_means(fold_performance_data, metric_name)
        rank_data = self.transpose_means(rank_data)

        # - Drop Virtual Best because it is oracle-like
        if (not keep_vb) and (
                (self.vb_name in rank_data.columns) or any(x in rank_data.columns for x in self.other_oracles)):
            rank_data = rank_data.drop(columns=[self.vb_name] + list(self.other_oracles))
        rank_data = rank_data.reset_index().drop(columns=["Dataset"])

        # - To avoid a bug following pivot rename of the column index header
        rank_data = pd.DataFrame(rank_data.values, columns=list(rank_data))

        if metric_maximize:
            # Descending means greater is better, hence maximize
            order = "descending"
        else:
            # Ascending mean  lower is better
            order = "ascending"

        # -- Check if data is "correct" such that statistical evaluation of autorank will be correct
        if rank_data.isnull().values.any():
            raise ValueError("Data for Autorank contains Nan values, this is not allowed!")

        # -- Run autorank
        result = autorank(rank_data, alpha=0.05, verbose=True, order=order, approach=approach)

        # -- Print Results
        latex_table(result)
        # Columns depend on the flow of autorank, for example, one could be:
        #   Mean Rank, Median, Mean absolute deviation, Confidence Interval, gamma, Magnitude
        #   Gamma is the effect size and magnitude a categorical discretization of it; it shows the difference between
        #       the first place and later placed algorithms.

        return result

    def wins_ties_losses(self, fold_performance_data, metric_name, metric_maximize):
        print("\n### Compute Wins/Ties/Losses (WTL) for metric {}...".format(metric_name))

        perf_data = fold_performance_data_to_fold_means(fold_performance_data, metric_name)
        perf_data = self.transpose_means(perf_data)

        if (self.vb_name in perf_data.columns) or any(x in perf_data.columns for x in self.other_oracles):
            perf_data = perf_data.drop(columns=[self.vb_name] + list(self.other_oracles))

        # -- Get overall wins/ties/losses
        if metric_maximize:
            winner_func = lambda s: s.max()
        else:
            winner_func = lambda s: s.min()

        def proc_func(x):
            highest_val_mask = x == winner_func(x)

            if sum(highest_val_mask) > 1:
                x[highest_val_mask] = "Tie"
            else:
                x[highest_val_mask] = "Win"

            x[~highest_val_mask] = "Loss"

            return x

        wtl_overview = perf_data.apply(proc_func, axis=1)

        # -- Prepare Print and Print Results
        res = {}
        for ens_tech in list(wtl_overview):
            res[ens_tech] = wtl_overview[ens_tech].value_counts(sort=False).to_dict()

            # Fill empty slots
            for key in ["Loss", "Tie", "Win"]:
                if key not in res[ens_tech]:
                    res[ens_tech][key] = 0
        sorted_by_wtl = sorted(list(wtl_overview), key=lambda x: (res[x]["Win"], res[x]["Tie"], res[x]["Loss"]),
                               reverse=True)
        print("## (W/T/L) - Ensemble Technique | For {} Ensemble Techniques across {} Datasets".format(
            len(sorted_by_wtl), len(wtl_overview)))
        print("".join("({}/{}/{}) - {}\n".format(res[x]["Win"], res[x]["Tie"], res[x]["Loss"], x) for x in
                      sorted_by_wtl))

    def simple_mean(self, fold_performance_data, metric_name, metric_maximize):
        """Take the mean over each fold. And for the folds' means, we compute the mean over all datasets for each
        ensemble technique. Which is then printed from best to worst.
        """

        print("\n### Compute Simple Mean over Techniques for metric {}...".format(metric_name))

        mean_result = fold_performance_data_to_fold_means(fold_performance_data, metric_name).groupby(
            ["Ensemble Technique"]).aggregate("mean").reset_index().sort_values(by=[metric_name],
                                                                                ascending=not metric_maximize)
        ordered_cols = mean_result[[metric_name, "Ensemble Technique"]]
        print(tabulate(ordered_cols, showindex=False, headers=[metric_name + " (1 = 100%)", "Ensemble Technique"]))

    def table_for_normalized_performance(self, fold_performance_data, metric_name, metric_maximize):

        perf_data = fold_performance_data_to_fold_means(fold_performance_data, metric_name)
        perf_data = self.transpose_means(perf_data)

        # Get First Row
        res_table = perf_data.apply(lambda x: "{} (\u00b1{})".format(round(x.mean(), self.n_rounds),
                                                                     round(x.std(), 2)),
                                    axis=0)
        res_table = pd.DataFrame(res_table).T
        ensembles = list(res_table)
        res_table["Data"] = pd.Series(["All Datasets"])
        res_table = res_table[["Data"] + ensembles]
        print(res_table.to_latex())

    # -- Other

    # ------ Some Evaler Ideas

    # -- Test impact of  complexity (in terms of data)
    #   Previous work state that less complex better for DS and more complex better for SB
    #       Can we show this here?
    #   Previous work states that not only complexity of data but the classification problem itself is relevant
    #       Can we show this here?

    # Impact of number of features

    # Impact of number of instances

    # Impact of number of categorical features

    # -- Test other
    # Impact of diversity in base model pool / Algorithm Performance complimentary

    # Impact of performance in base model pool (e.g. mean accuracy)
    # Algorithm Performance Variability

    # Distribution of best algorithms (average wins%)


def fold_performance_data_to_fold_means(fold_performance_data, metric_name):
    return fold_performance_data.drop(columns=["Fold"]).groupby(
        by=["Dataset", "Ensemble Technique"]).aggregate("mean").reset_index()[
        ["Dataset", "Ensemble Technique", metric_name]]
