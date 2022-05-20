import numpy as np
import pytest
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import check_is_fitted, NotFittedError

from results.data_utils import get_default_preprocessing
from assembled.compatibility.faked_classifier import FakedClassifier
from test.assembled_metatask_util import build_multiple_test_classification_metatasks

metatasks = build_multiple_test_classification_metatasks()


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

    def test_faked_classifier_init(self):
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=4, random_state=0, n_classes=3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        clf = DummyClassifier()
        clf.fit(X_train, y_train)

        fc = FakedClassifier(X_test, clf.predict(X_test), clf.predict_proba(X_test))

        assert hasattr(fc, "predictions_")
        assert hasattr(fc, "confidences_")
        assert isinstance(fc.predictions_, np.ndarray)
        assert isinstance(fc.confidences_, np.ndarray)
        assert fc.predictions_.shape == (len(y_test),)
        assert fc.confidences_.shape == (len(y_test), 3)

    def test_faked_classifier_predict_with_validation_data(self):
        X, y = make_classification(n_samples=10000, n_features=10, n_informative=4, random_state=0, n_classes=3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        clf = DummyClassifier()
        val_confs = cross_val_predict(clf, X_train, y_train, cv=5, method="predict_proba")
        clf.fit(X_train, y_train)
        val_preds = clf.classes_.take(np.argmax(val_confs, axis=1), axis=0)
        preds = clf.predict(X_test)
        confs = clf.predict_proba(X_test)

        # Concat as required
        base_model_train_X = base_model_test_X = np.vstack((X_train, X_test))
        base_model_known_predictions = np.hstack((val_preds, preds))
        base_model_known_confidences = np.vstack((val_confs, confs))
        base_model_train_y = np.hstack((y_train, y_test))

        fc = FakedClassifier(base_model_test_X, base_model_known_predictions, base_model_known_confidences)

        fc.fit(base_model_train_X, base_model_train_y)

        assert np.array_equal(fc.predict(X_test), preds)
        assert np.array_equal(fc.predict_proba(X_test), confs)
        assert np.array_equal(fc.predict(X_train), val_preds)
        assert np.array_equal(fc.predict_proba(X_train), val_confs)

    def test_faked_classifier_fit(self):
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=4, random_state=0, n_classes=3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        clf = DummyClassifier()
        st = time.time()
        clf.fit(X_train, y_train)
        fit_time = time.time() - st

        fc = FakedClassifier(X_test, clf.predict(X_test), clf.predict_proba(X_test))
        fc.fit(X_train, y_train)

        assert check_is_fitted(fc, ["le_", "classes_"]) is None

        fc = FakedClassifier(X_test, clf.predict(X_test), clf.predict_proba(X_test), simulate_time=True,
                             fit_time_=fit_time)
        st = time.time()
        fc.fit(X_train, y_train)
        sim_fit_time = time.time() - st

        assert fit_time <= sim_fit_time

    def test_faked_classifier_predict_and_predict_proba(self):
        # Get and Fit data for Fake Classifier
        X, y = make_classification(n_samples=1000, n_features=10, n_informative=4, random_state=0, n_classes=3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        clf = DummyClassifier()
        clf.fit(X_train, y_train)

        # Get Predictions and timings
        st = time.time()
        preds = clf.predict(X_test)
        pred_time = time.time() - st
        st = time.time()
        confs = clf.predict_proba(X_test)
        confs_time = time.time() - st

        fc = FakedClassifier(X_test, preds, confs)

        # Not fitted?
        with pytest.raises(NotFittedError):
            fc.predict(X_test)

        # Fitted but broken preds are returned?
        fc.fit(X_train, y_train)
        assert np.array_equal(fc.predict(X_test), preds)
        assert np.array_equal(fc.predict_proba(X_test), confs)

        # Timings are correct?
        fc = FakedClassifier(X_test, preds, confs, simulate_time=True, predict_time_=pred_time,
                             predict_proba_time_=confs_time)
        fc.fit(X_train, y_train)
        st = time.time()
        fc.predict(X_test)
        sim_pred_time = time.time() - st
        st = time.time()
        fc.predict_proba(X_test)
        sim_confs_time = time.time() - st

        assert pred_time <= sim_pred_time
        assert confs_time <= sim_confs_time
