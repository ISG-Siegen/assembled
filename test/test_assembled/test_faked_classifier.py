from assembledopenml.openml_assembler import OpenMLAssembler
from results.data_utils import get_default_preprocessing

import numpy as np

# --- Get Metatasks used for testing
omla = OpenMLAssembler(openml_metric_name="area_under_roc_curve", maximize_metric=True, nr_base_models=5)
metatasks = []
for task_id in [3913, 3560, 9957, 23]:
    # Build meta-dataset for each task
    meta_task = omla.run(task_id)
    meta_task.filter_predictors()
    metatasks.append(meta_task)


class TestFakedClassifier:

    def test_full_faked_classifier_predictions_after_init(self):
        pre_proc = get_default_preprocessing()

        for mt in metatasks:

            for base_models, X_meta_train, X_meta_test, y_meta_train, y_meta_test, assert_meta_train_pred, \
                assert_meta_test_pred, assert_meta_train_conf, assert_meta_test_conf, X_train, y_train \
                    in mt._exp_yield_evaluation_data_across_folds(0.5, 0, True, True, False, pre_proc,
                                                                  include_test_data=True):

                for bm_name, bm in base_models:
                    assert np.array_equal(assert_meta_train_pred[bm_name].to_numpy(), bm.predict(X_meta_train))
                    assert np.array_equal(assert_meta_test_pred[bm_name].to_numpy(), bm.predict(X_meta_test))
                    assert np.array_equal(assert_meta_train_conf[["confidence.{}.{}".format(class_name, bm_name)
                                                                  for class_name in np.unique(y_train)]].to_numpy(),
                                          bm.predict_proba(X_meta_train))
                    assert np.array_equal(assert_meta_test_conf[["confidence.{}.{}".format(class_name, bm_name)
                                                                 for class_name in np.unique(y_train)]].to_numpy(),
                                          bm.predict_proba(X_meta_test))

                    # TODO would need to write own check as data checks need specific fake data during init time
                    #   which fake models can not handle. But other checks work.
                    # check_estimator(FakedClassifier(i_X_test, i_y_train, model_predictions, model_confidences))
