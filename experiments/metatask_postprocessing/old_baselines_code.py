import numpy as np
import pandas as pd
from typing import Optional, Callable


class VirtualBestAlgorithm:
    def __init__(self, base_est_predictions: pd.DataFrame, ground_truth: pd.Series,
                 classification: bool = True, stack_method: str = "predict_proba",
                 base_est_confidences: Optional[pd.DataFrame] = None, conf_cols_to_est_label: Optional[dict] = None):
        """Simulated Oracle-like Selector that can return predictions of the Virtual Best Algorithm (VBA)

        Parameters
        ----------
        base_est_predictions : pd.DataFrame
            Predictions of base models. Each row represent an instance and the column a base model's predictions.
        ground_truth : pd.Series
            The ground truth of the instances.
        classification : bool, default=True
            Whether the VBA is used for a classification or regression task.
        stack_method : {"predict_proba", "predict"}, default="predict_proba"
            Use predictions or the confidences of predictions to determine the the set of best example_algorithms if no correct
             classification/regression has happend for any algorithm.
        base_est_confidences: pd.DataFrame, default=None
            Confidences of the base models for each label. Required for stack_method="predict_proba".
        conf_cols_to_est_label: dict, default=None
            A dict of dicts detailing for each predictor name (column names of base_est_predictions) the relationship
            between the class labels and the column names of confidences (column names of base_est_confidences).
        """
        self.base_est_predictions = base_est_predictions
        self.ground_truth = ground_truth
        self.classification = classification
        self.base_est_confidences = base_est_confidences
        self.conf_cols_to_est_label = conf_cols_to_est_label

        self.stack_method = stack_method

        # -- Validate
        if stack_method not in ["predict_proba", "predict"]:
            raise ValueError("Unknown/not-supported stack method: {}".format(stack_method))
        if self.stack_method == "predict_proba" and (base_est_confidences is None):
            raise ValueError("Require confidence values to simulate predict proba usage.")

        # -- Shortcuts
        self.performance_cols = list(self.base_est_predictions)
        self.class_labels = list(list(self.conf_cols_to_est_label.items())[0][1].keys())

    def _handle_no_vba(self, input_tuple: tuple) -> list:
        """Handle the case that no base model was able to classify the current instance correctly"""

        idx = input_tuple[0]
        best_ind_set = input_tuple[1]

        # Skip if instance was correctly classified at least once
        if best_ind_set:
            return best_ind_set

        # Otherwise handle it
        if self.stack_method == "predict":
            # If stacked_method is predict, any algorithm is the virtual best.
            #   !This ignores error values or similar, because we dont have/use them for this stack method!
            return list(range(self.base_est_predictions.shape[1]))
        else:
            # Predict proba case, here the algorithm with the highest confidence for the correct label
            #  for this specific instance is the best algorithm.

            # Find the correct label for this specific instance
            rel_class_label = self.ground_truth[idx]
            # Use this to find the correct confidence columns
            rel_cols = [self.conf_cols_to_est_label[pred_name][rel_class_label]
                        for pred_name in self.performance_cols]
            # Ues this and the instance index to find the correct confidence values
            instance_confs = self.base_est_confidences.loc[idx, rel_cols]
            # Use this to find the algorithm indices with maximal confidence
            return np.argwhere(np.array(instance_confs) == np.max(np.array(instance_confs))).flatten().tolist()

    @property
    def virtual_best_predictors_indices_sets(self) -> pd.Series:
        """Create a list of lists (stored using a pd.Series) that contains the set of best example_algorithms' indices
         for each instance

        For classification, we assume that an algorithm is a "best" algorithm for an instance if it can correctly
        classify the current instance. The probability of correct classification is not relevant for this.
        """

        if self.classification:
            best_predictors_set = [
                np.argwhere(np.array(preds_list) == np.max(np.array(preds_list))).flatten().tolist() for preds_list in
                self.base_est_predictions.apply(lambda x: x == self.ground_truth, axis=0).astype(int).values.tolist()]
            instance_indices = self.base_est_predictions.index.tolist()

            # Transform to tuple and then series for post processing
            best_predictors_set = list(zip(instance_indices, best_predictors_set))
            best_predictors_set = pd.Series(best_predictors_set, index=instance_indices)

            # If for an instance in classification no example_algorithms was able to correctly classify the instance,
            #   we need to handle this.
            best_predictors_set = best_predictors_set.apply(self._handle_no_vba)

        else:
            raise NotImplementedError("Regression not supported yet.")

        return best_predictors_set

    @property
    def virtual_best_predictors_indices(self) -> pd.Series:
        """Get the selection vector of the VBA"""

        # We assume that each instance has an predictor index set associated, hence we just take the first one
        #   everytime.
        return self.virtual_best_predictors_indices_sets.apply(lambda x: x[0])

    def predict(self) -> np.ndarray:
        """Predict using the VBA by returning the selected predictions

        Returns
        -------
        p : ndarray of shape (n_samples,)
            The predicted value of the base models selected by the VBA.
        """
        return self.base_est_predictions.to_numpy()[np.arange(len(self.base_est_predictions)),
                                                    self.virtual_best_predictors_indices]


class SingleBestAlgorithm:
    def __init__(self, metric_to_use: Callable, maximize_metric: bool = True):
        """Simulated Single Best Algorithm Selector that can returns predictions

        We find the SBA based on the predictions of a model and not its confidences!
        This can make a large difference depending on the used metric.

        Parameters
        ----------
        metric_to_use : openml metric function, default=None
            The metric function that should be used to determine the single best algorithm
            Special format required due to OpenML's metrics.
        maximize_metric : bool, default=True
            Whether the metric computed by the metric function passed by metric_to_use is to be maximized or not.
        """

        self.metric_to_use = metric_to_use
        self.maximize_metric = maximize_metric
        self.best_algo_index = None
        self.selected_indices = None

    def _find_single_best_algorithm(self, base_est_predictions, y):
        """Find the index of the best algorithm"""
        # -- Select best algorithm based on metric (not error as it is biased)
        algo_metric_perf = base_est_predictions.apply(lambda x: self.metric_to_use(y, x),
                                                      axis=0).to_numpy()

        if self.maximize_metric:
            self.best_algo_index = np.argmax(algo_metric_perf)
        else:
            self.best_algo_index = np.argmin(algo_metric_perf)

    def fit(self, base_est_predictions: pd.DataFrame, y: pd.Series):
        """Find the single best algorithm and store it for later

        Parameters
        ----------
        base_est_predictions : pd.DataFrame
            Training Predictions of base models.
            Each row represent an instance and the column a base model's predictions.
        y : pd.Series, array-like
            ground-truth
        """
        self._find_single_best_algorithm(base_est_predictions, y)

    def predict(self, X: pd.DataFrame, base_est_predictions: pd.DataFrame) -> np.ndarray:
        """

        Parameters
        ----------
        X : {array-like, pd.DataFrame}
            Test input samples
        base_est_predictions : pd.DataFrame
            Test Predictions of base models.
            Each row represent an instance and the column a base model's predictions.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The prediction values of the single best algorithm
        """
        self.selected_indices = np.full(X.shape[0], self.best_algo_index)
        return base_est_predictions.to_numpy()[np.arange(len(base_est_predictions)), self.selected_indices]
