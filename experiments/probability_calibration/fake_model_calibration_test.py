from assembled.metatask import MetaTask
from results.data_utils import get_valid_benchmark_ids, get_default_preprocessing

from experiments.probability_calibration.calibration_stuff import calibration_curves

from sklearn.calibration import CalibratedClassifierCV

# -- Main
if __name__ == "__main__":

    # --- Input para
    valid_task_ids = get_valid_benchmark_ids(base_path="../../results/benchmark_metatasks")
    test_split_frac = 0.5
    rng_seed = 3151278530
    test_split_rng = 581640921

    # --- Get Overview over calibration data
    for task_nr, task_id in enumerate(valid_task_ids, start=1):
        mt = MetaTask()
        mt.read_metatask_from_files("../../results/benchmark_metatasks", task_id)
        print("#### Process Task {} for dataset {} ({}/{}) ####".format(mt.openml_task_id, mt.dataset_name,
                                                                        task_nr, len(valid_task_ids)))

        # -- Fake call to get base model and data for tests below
        base_models, X_test, y_test = mt._exp_get_base_models_for_all_folds(preprocessor=get_default_preprocessing(),
                                                                            pre_fit_base_models=True,
                                                                            base_models_with_names=True)
        calibration_curves(base_models, X_test, y_test, mt)

        # -- Just for one fold
        for base_models, X_meta_train, X_meta_test, y_meta_train, y_meta_test in mt._exp_yield_evaluation_data_across_folds(
                test_split_frac, test_split_rng, preprocessor=get_default_preprocessing(),
                pre_fit_base_models=True, base_models_with_names=True, label_encoder=False):

            calibration_curves(base_models, X_meta_test, y_meta_test, mt, save=False, show_plot=True)

            cal_base_models = []
            for name, bm in base_models:
                # -- Calibrate
                cal_bm = CalibratedClassifierCV(bm, method="sigmoid", cv="prefit")
                cal_bm.fit(X_meta_train, y_meta_train)
                cal_bm.le_ = bm.le_
                cal_base_models.append((name, cal_bm))

            calibration_curves(cal_base_models, X_meta_test, y_meta_test, mt, save=False, show_plot=True,
                               plt_prefix="Calibrated ")
            break
