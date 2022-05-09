import pandas as pd
from autorank import autorank, latex_table


class Evaler:
    """Evaler enables easy access to different evaluations methods

    Parameters
    ----------
    vb_name
    sb_name
    """

    def __init__(self, vb_name="custom.VirtualBest", sb_name="custom.SingleBest"):
        self.vb_name = vb_name
        self.sb_name = sb_name

    def use_autorank(self, fold_performance_data, metric_name, metric_maximize, approach="frequentist"):
        print("\n### Use Autorank to produce evaluations for metric {} and approach {}...".format(metric_name,
                                                                                                  approach))

        # --- Transform fold_performance_data into data for autorank
        rank_data = fold_performance_data.drop(columns=["Fold"]).groupby(
            by=["Dataset", "Ensemble Technique"]).aggregate("mean").reset_index()[
            ["Dataset", "Ensemble Technique", metric_name]]
        rank_data = rank_data.pivot(index="Dataset", columns="Ensemble Technique")
        rank_data.columns = rank_data.columns.droplevel(0)

        # - Drop Virtual Best because it is oracle-like
        rank_data = rank_data.drop(columns=[self.vb_name]).reset_index().drop(columns=["Dataset"])

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
