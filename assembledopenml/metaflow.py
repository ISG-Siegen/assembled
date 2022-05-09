from openml.runs import get_run as openml_get_run
from openml.flows import get_flow as openml_get_flow
from assembledopenml.util.data_fetching import fetch_arff_file_to_dataframe
from assembledopenml.util.common import make_equal
from typing import Union, List


class MetaFlow:
    def __init__(self, flow_id: int, description: Union[str, None], best_performance: Union[str, None],
                 best_run_id: int, conf_tol: float = 0.01):
        """Internal Object of a Flow filled with (for us relevant) metatasks about the flow.

        Parameters
        ----------
        flow_id : int
            The ID of the flow for which we want to build a metaflow.
        description: Union[str, None]
            The description (or name) of the flow.
        best_performance: Union[str, None]
            The best performance of a run for the flow.
        best_run_id: int
            The id of best performing run of the flow.
        conf_tol: float
            The tolerance for incorrect confidence-predictions. Determines in how far we think numerical precision is
            the reason for a wrong confidence value.
        """
        self.flow_id = flow_id
        self.description = description
        self.best_performance = best_performance
        self.best_run_id = best_run_id
        self.conf_tol = conf_tol

        # -- init later
        self.predictions_url = None
        self.conf_prefix = None
        self.predictions = None
        self.confidences = None
        self.file_ground_truth = None
        self.name = None

        # -- flags
        self.file_ground_truth_corrupted = False
        self.confidences_corrupted = False
        self.confidences_fixed = False

        # -- Post Checks
        # Automatically fill description if it is None
        if self.description is None:
            self.description = openml_get_flow(self.flow_id).name

    @property
    def is_bad_flow(self):
        return self.confidences_corrupted and (not self.confidences_fixed)

    @property
    def corruption_details(self):
        return {
            "file_ground_truth_corrupted": self.file_ground_truth_corrupted,
            "confidences_corrupted": self.confidences_corrupted,
            "confidences_fixed": self.confidences_fixed
        }

    def _parse_label_col(self, predictions_data):
        """Read Label Column from Predictions file"""
        known_label_names = ["correct", "truth"]

        # Find label column in data
        for label_name in known_label_names:
            if label_name in list(predictions_data):
                return label_name

        # The case if we did not return (= the case were no known label is in columns)
        raise RuntimeError("Unknown predictions-like file format. ",
                           "Unable to parse label column for: {} ".format(self.predictions_url))

    def _parse_conf_cols(self, predictions_data):
        # Get columns with confidence values
        known_confidence_prefixes = ["confidence."]

        confidence_cols = []
        hit = False
        # Collect conf columns and prefix is in data
        for conf_prefix in known_confidence_prefixes:
            for col_name in list(predictions_data):
                if conf_prefix in col_name:
                    confidence_cols.append(col_name)
                    hit = True
            if hit:
                return conf_prefix, confidence_cols

        raise RuntimeError("Unknown predictions-like file format. ",
                           "Unable to parse confidence columns for: {} ".format(self.predictions_url))

    def _confidence_to_predictions(self):
        # Assumption: highest confidence equals prediction
        return self.confidences.idxmax(axis=1).apply(lambda x: x[len(self.conf_prefix):])

    def _gather_wrong_confidences(self, conf_preds):
        wrong_confs_mask = conf_preds != self.predictions

        # Get relevant subsets
        pred_wrong = self.predictions[wrong_confs_mask]
        conf_pred_wrong = conf_preds[wrong_confs_mask]
        confs_wrong = self.confidences[wrong_confs_mask]

        # Values to check
        tol_equal = True
        not_equal_at_all_idx = []
        equal_with_tol_idx = []

        # Check relevant confidences
        for idx, pred, conf_pred in zip(pred_wrong.index, pred_wrong, conf_pred_wrong):
            conf_of_pred = confs_wrong.loc[idx, "{}{}".format(self.conf_prefix, pred)]
            conf_of_conf_pred = confs_wrong.loc[idx, "{}{}".format(self.conf_prefix, conf_pred)]

            # Check if equal
            if conf_of_conf_pred != conf_of_pred:
                # If not, check if equal within tolerance
                if not ((conf_of_pred - self.conf_tol) < conf_of_conf_pred < (conf_of_pred + self.conf_tol)):
                    tol_equal = False  # even with tolerance, the confidence are not equal
                    not_equal_at_all_idx.append((idx, conf_of_pred))
                else:
                    equal_with_tol_idx.append((idx, conf_of_pred))

        return tol_equal, not_equal_at_all_idx, equal_with_tol_idx

    def _validate_confidences(self):
        """
            Check if the confidences accurately reflect the predictions
            Return True if valid, False if not
        """
        # FIXME, might want to add a check here that all values sum up to 1

        conf_preds = self._confidence_to_predictions()

        # Check if confidences are correct
        if conf_preds.equals(self.predictions):
            return

        # Class-predictions and confidence-predictions are not identical.
        self.confidences_corrupted = True

        # Gather data to check if we can fix the corrupted confidences
        tol_equal, not_equal_at_all_idx, equal_with_tol_idx = self._gather_wrong_confidences(conf_preds)

        # Check if it is fixable
        if tol_equal:
            # Predictions are not identical because the confidence is almost equal.
            # We assume this is an artifact from randomness in classifiers or precision and ignore it.
            # We, however, update the values to make them equal for later usage.

            # Make Data Equal
            for idx, conf_of_pred in equal_with_tol_idx:
                self.confidences.loc[idx] = make_equal(conf_of_pred, self.confidences.loc[idx])
            self.confidences_fixed = True

        else:
            # ----- The Problematic Cases

            # -- Check if it is a model were probabilities/confidence score is not necessarily representative
            #        of the prediction due to too small datasets or cross validation based proba-calculation.
            not_rep_conf = [
                # small dataset or cross validation make confidence (proba) bad,
                # see https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
                "classifier=sklearn.svm.classes.SVC)",
                "svc=sklearn.svm.classes.SVC)",
                "sklearn.svm._classes.SVC)",

                # Special Case for Adaboost, which can have inaccurate probability results for small datasets
                # see ,e.g., Adaboost https://stats.stackexchange.com/questions/329080/adaboost-probabilities
                "classifier=sklearn.ensemble.weight_boosting.AdaBoostClassifier("

            ]
            if any(x in self.description for x in not_rep_conf):
                # The confidence for the predictions is not representative because of the model's rng.
                # To solve this, we equalize all too high confidences and the prediction's confidence
                to_make_equal = equal_with_tol_idx + not_equal_at_all_idx
                for idx, conf_of_pred in to_make_equal:
                    self.confidences.loc[idx] = make_equal(conf_of_pred, self.confidences.loc[idx])
                self.confidences_fixed = True

            else:
                # FIXME UNKNOWN REASON FOR BAD FLOW; MIGHT REQUIRE POSTPROCESSING; STORE FOR LATER BUT IGNORE FOR NOW
                # Potential Reasons
                #   Data uploaded to OpenML is Wrong
                #   Malicious content
                #   Bugs in OpenML Backend or ML Tool Pipeline
                pass

    def _validate_predictions(self, class_labels):
        # -- Check Format of Pred col (make sure it has the same string style as the ground truth col)
        if set(self.predictions.unique().tolist()) - set(class_labels):
            # TODO handle this as a special case?
            raise ValueError("The Prediction Column for Flow {} has the wrong label-name format: {}".format(
                self.flow_id, self.predictions.unique().tolist()))

    def get_predictions_data(self, class_labels: List[str]):
        """Fill the metaflow object with predictions data (and other relevant variables)

        Parameters
        ----------
        class_labels: List[str]
            The names of each class label
        """
        # -- Load Predictions file
        self.predictions_url = openml_get_run(self.best_run_id).predictions_url
        predictions_data, _ = fetch_arff_file_to_dataframe(self.predictions_url)

        # -- Parse Predictions data
        # FIXME we are assuming a default format here. If this is different, we can not work with it and it will crash.

        # - Parse y_true_col_name from data file
        y_true_col_name = self._parse_label_col(predictions_data)

        # - Get predictions in correct format (str decode) and in order of instances (sort_values + reset index)
        predictions_data = predictions_data.sort_values(by="row_id").reset_index()
        self.predictions = predictions_data["prediction"].str.decode("utf-8")
        self.file_ground_truth = predictions_data[y_true_col_name].str.decode("utf-8")
        self._validate_predictions(class_labels)

        # - Parse confidence columns
        self.conf_prefix, conf_cols = self._parse_conf_cols(predictions_data)
        if (len(conf_cols) % len(class_labels)) != 0:
            raise ValueError("Too few confidence columns found in predictions file: {} ".format(self.predictions_url),
                             "Expected {} cols for each predictor. ".format(len(class_labels)),
                             "Found at least 1 predictior with too few columns.")

        # - Get confidence values
        self.confidences = predictions_data[[self.conf_prefix + n for n in class_labels]].copy()
        self._validate_confidences()
