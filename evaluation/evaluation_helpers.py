import seaborn as sb
import matplotlib as mp


def draw_stripplot(df, x, y, hue,
                   hue_order=None,
                   xlabel=None, ylabel=None, y_labels=None, title=None,
                   legend_title=None, legend_loc='best', legend_labels=None, size=None):
    """Please be aware: The code for stripplot visualization is adapted from the AutoMl Benchmark!

    For the original code,
    please see: https://github.com/openml/automlbenchmark/blob/master/amlb_report/visualizations/stripplot.py
    """
    colormap = 'colorblind'

    with sb.axes_style('whitegrid', rc={'grid.linestyle': 'dotted'}), sb.plotting_context('paper'):

        # Initialize the figure
        strip_fig, axes = mp.pyplot.subplots(dpi=120, figsize=(10, len(df.index.unique())))
        axes.set_xscale('linear')

        sb.despine(bottom=True, left=True)

        # Show each observation with a scatterplot
        sb.stripplot(data=df,
                     x=x, y=y, hue=hue,
                     hue_order=hue_order,
                     palette=colormap,
                     dodge=True, jitter=False,
                     alpha=.25, zorder=1)

        # Show the conditional means
        sb.pointplot(data=df,
                     x=x, y=y, hue=hue,
                     hue_order=hue_order,
                     palette=colormap,
                     dodge=0.68, join=False,
                     markers='d', scale=.75, ci=None)

        # Improve the legend
        handles, labels = axes.get_legend_handles_labels()
        dist = int(len(labels) / 2)
        handles, labels = handles[dist:], labels[dist:]
        if legend_labels is not None:
            if isinstance(legend_labels, list):
                labels = legend_labels
            else:
                labels = map(legend_labels, labels)

        axes.legend(handles, labels, title=legend_title or hue,
                    handletextpad=0, columnspacing=1,
                    loc=legend_loc, ncol=1, frameon=True)

        axes.set_title('' if not title else title, fontsize='xx-large')
        axes.set_xlabel('' if not xlabel else xlabel, fontsize='x-large')
        axes.set_ylabel('' if not ylabel else ylabel, fontsize='x-large')
        axes.tick_params(axis='x', labelsize='x-large')
        axes.tick_params(axis='y', labelsize='x-large')

        if y_labels is not None:
            axes.set_yticklabels(y_labels)

        return strip_fig
