import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibrationDisplay


def calibration_curves(base_models, X_test, y_test, mt, interactive=False, save=True, show_plot=False, plt_prefix=""):
    """Code adapted from: https://scikit-learn.org/stable/auto_examples/calibration/plot_compare_calibration.html
    """

    if interactive:
        import matplotlib as mpl
        mpl.use('TkAgg')

    # Get base model name data
    inv_descriptions = {v: k for k, v in mt.predictor_descriptions.items()}
    counts_dict = Counter([mt.predictor_descriptions[bm_name] for bm_name, _ in base_models])
    counts_sorted = sorted(list(counts_dict.items()), key=lambda x: x[1], reverse=True)

    # Build Figure
    fig = plt.figure(figsize=(20, 8))
    gs = GridSpec(4, 2)
    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}
    colors = plt.cm.get_cmap("Dark2")

    if mt.n_classes <= 2:
        # Binary class case
        add_to_name = ""
        add_to_title = ""

        # Get calibration data from fake model as is
        for i, (name, clf) in enumerate(base_models):
            display = CalibrationDisplay.from_estimator(clf, X_test, y_test, n_bins=10, name=name,
                                                        ax=ax_calibration_curve)
            calibration_displays[name] = display

    else:
        # Multiclass case
        add_to_name = "_all_c"
        add_to_title = "Multiclass [{}] - ".format(mt.n_classes)

        # Get calibration data from base model but treat each class one vs rest style.
        # Here, we build one large y_test and y_prob that includes the distribution of the classifier for each
        #   class.
        for i, (name, clf) in enumerate(base_models):
            y_prob = clf.predict_proba(X_test)
            mod_y_prob_list = []
            mod_y_test_list = []
            for i_class, class_name in enumerate(clf.classes_):
                # Select class proba
                mod_y_prob = y_prob[:, i_class]
                # Transform y to "positive" class array
                mod_y_test = clf.le_.transform(y_test)
                cl_mask = mod_y_test == i_class
                mod_y_test[cl_mask] = 1
                mod_y_test[~cl_mask] = 0
                mod_y_prob_list.append(mod_y_prob)
                mod_y_test_list.append(mod_y_test)

            mod_y_prob = np.array(mod_y_prob_list).flatten()
            mod_y_test = np.array(mod_y_test_list).flatten()
            mod_name = name + add_to_name

            display = CalibrationDisplay.from_predictions(mod_y_test, mod_y_prob, n_bins=10, name=mod_name,
                                                          ax=ax_calibration_curve,
                                                          color=colors(i % len(colors.colors)))
            calibration_displays[mod_name] = display

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title(
        "{}{}Calibration plots [{} - {} - Test_n_instances: {}]".format(plt_prefix, add_to_title,
                                                                        mt.openml_task_id,
                                                                        mt.dataset_name,
                                                                        len(y_test)))
    # Remove legend because it does not work with too many base models
    ax_calibration_curve.get_legend().remove()

    # Build lower part of the plot for the top most often occurring base models
    grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
    for i, (n, count) in enumerate(counts_sorted[:4]):
        name = inv_descriptions[n] + add_to_name
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])

        ax.hist(
            calibration_displays[name].y_prob,
            range=(0, 1),
            bins=20,
            label=name,
            color=colors(i % len(colors.colors))
        )

        ax.set(title="{} | Count: {}".format(n[-40:], count), xlabel="Mean predicted probability",
               ylabel="Count")

    plt.tight_layout()

    if save:
        plt.savefig("./output/calibration_curves_for_metatask_{}.pdf".format(mt.openml_task_id))

    if interactive or show_plot:
        plt.show()
