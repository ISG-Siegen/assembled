from results.data_utils import get_valid_benchmark_ids
from experiments.assembledevaler.resulter import Resulter
from experiments.assembledevaler.ploter import Ploter
from experiments.assembledevaler.evaler import Evaler
from ensemble_techniques.util.metrics import OpenMLAUROC

if __name__ == "__main__":
    # Setup analysis and eval tools
    reser = Resulter(get_valid_benchmark_ids("../results/openml_benchmark/benchmark_metatasks"),
                     path_to_metatask="../results/openml_benchmark/benchmark_metatasks",
                     path_to_benchmark_output="../results/openml_benchmark/benchmark_output")
    plter = Ploter(path_to_plt_output="../results/openml_benchmark/evaluation")
    evler = Evaler()
    metric = OpenMLAUROC
    metric_name = metric.name
    metric_maximize = metric.maximize

    # -- Get Data
    fp_data = reser.get_fold_performance_data(metric)
    fp_data = reser.get_vb_sb_normalization(fp_data, metric_name=metric_name, vb_name=evler.vb_name,
                                            sb_name=evler.sb_name)  # adds column with name: "VB_SB_GAP"
    fp_data = reser.get_relative_improvement_to_sb(fp_data, metric_name=metric_name,
                                                   sb_name=evler.sb_name)  # adds column "RI_SB"

    mt_metadata = reser.get_metatask_data(["n_features", "n_instances", "n_classes"])
    plter.instance_space_plot(fp_data[fp_data["Ensemble Technique"] != evler.vb_name], metric_name, mt_metadata)

    # -- Optional Preprocessing
    # fp_data = reser.filter_fp_data(fp_data, [3913, 3917, 10101])

    # -- Some Basic Plots
    # plter.stripplot(fp_data, metric_to_plot=metric_name)
    # plter.stripplot(fp_data, metric_to_plot=metric_name, split=True)

    # ---- Finding the Average Best Method
    # -- Wins / Ties / Losses
    evler.wins_ties_losses(fp_data, metric_name, metric_maximize=True)

    # -- Mean
    evler.simple_mean(fp_data, "RI_SB", metric_maximize=True)

    # -- Overall Ranking Stuff
    autorank_results = evler.use_autorank(fp_data, metric_name, metric_maximize=False)
    plter.autorank_plot(autorank_results, ext="_{}".format(metric.name))

    # fixes a bug with autorank by converting it to a minimization problem
    anti_bug_data = fp_data.copy()
    anti_bug_data["RI_SB"] *= -1
    autorank_results = evler.use_autorank(anti_bug_data, "RI_SB", metric_maximize=False, keep_vb=False)
    plter.autorank_plot(autorank_results, ext="_{}".format("RI_SB"))
