import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from typing import Optional, Union, Callable
from collections import Counter


class SimulatedStackingClassifier:
    """StackingClassifier like sklearn's StackingClassifier but without having to train/evaluate base models first"""

    def __init__(self, final_estimator: BaseEstimator, passthrough: bool = False,
                 stack_method: str = "predict_proba", conf_cols_to_est_label: Optional[dict] = None):
        """Init

        Parameters
        ----------
        final_estimator : sklearn estimator/pipeline
            Level-1 Meta-Estimator used for stacking.
        passthrough : str, default=False
            Whether to let the Meta-Estimator utilize the original training features or not.
        stack_method : str, default="predict_proba"
            Used stacking method. That is, what type of input is passed to the Meta-Estimator from the base models.
        conf_cols_to_est_label: dict, default=None
            A dict of dicts detailing for each predictor name (column names of base_est_predictions) the relationship
            between the class labels and the column names of confidences (column names of base_est_confidences).
        """
        self.final_estimator = final_estimator
        self.passthrough = passthrough
        self.stack_method = stack_method
        self.conf_cols_to_est_label = conf_cols_to_est_label

        # -- Validate
        if stack_method not in ["predict_proba", "predict"]:
            raise ValueError("Unknown/not-supported stack method: {}".format(stack_method))

    def _simulate_and_build_X(self, X_features, base_est_predictions, base_est_confidences=None):
        # To use or not use confidences / probabilities
        if self.stack_method == "predict_proba":
            X_predictions = self._preprocess_proba(base_est_confidences)
        else:
            X_predictions = self._preprocess_predictions(base_est_predictions)

        # pass through yes/no
        if self.passthrough:
            # yes
            return pd.concat([X_features, X_predictions], axis=1)

        # no
        return X_predictions

    def _preprocess_proba(self, proba_input):
        if self.conf_cols_to_est_label is None:
            # Did not pass dict to filter cols
            return proba_input

        # If binary prediction case, drop always first label
        class_labels = list(list(self.conf_cols_to_est_label.items())[0][1].keys())
        if len(class_labels) == 2:
            second_label = class_labels[1]
            cols_w_o_first_label = [vals[second_label] for key, vals in self.conf_cols_to_est_label.items()]
            proba_input = proba_input[cols_w_o_first_label]

        return proba_input

    @staticmethod
    def _preprocess_predictions(predictions_input):
        return predictions_input.apply(lambda x: x.cat.codes, axis=1)

    def fit(self, X: pd.DataFrame, y: pd.Series, base_est_predictions: pd.DataFrame,
            base_est_confidences: Optional[pd.DataFrame] = None):
        """Train the Meta-Estimator

        Parameters
        ----------
        X : {array-like, pd.DataFrame}
            Training Input Samples
        y : pd.Series, array-like
            Training ground-truth
        base_est_predictions : pd.DataFrame
            Training Predictions of base models.
            Each row represent an instance and the column a base model's predictions.
        base_est_confidences: pd.DataFrame, default=None
            Confidences of the base models for each label. Required for default="predict_proba".
        """

        if self.stack_method == "predict_proba" and (base_est_confidences is None):
            raise ValueError("Require confidence values to simulate predict proba usage.")

        fit_X = self._simulate_and_build_X(X, base_est_predictions, base_est_confidences=base_est_confidences)
        self.final_estimator.fit(fit_X, y)

    def predict(self, X: pd.DataFrame, base_est_predictions: pd.DataFrame,
                base_est_confidences: Optional[pd.DataFrame] = None) -> np.ndarray:
        """

        Parameters
        ----------
        X : {array-like, pd.DataFrame}
            Test input samples
        base_est_predictions : pd.DataFrame
            Test Predictions of base models.
            Each row represent an instance and the column a base model's predictions.
        base_est_confidences: pd.DataFrame, default=None
            Test confidences of the base models for each label. Required for stack_method="predict_proba".

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predictions of the meta-estimator
        """
        if self.stack_method == "predict_proba" and (base_est_confidences is None):
            raise ValueError("Require confidence values to simulate predict proba usage.")

        pred_X = self._simulate_and_build_X(X, base_est_predictions, base_est_confidences=base_est_confidences)
        return self.final_estimator.predict(pred_X)


class SimulatedVotingClassifier:
    """VotinggClassifier like sklearn's VotinggClassifier but without having to train/evaluate base models first"""

    def __init__(self, voting: str = "hard", conf_cols_to_est_label: Optional[dict] = None):
        """

        Parameters
        ----------
        voting : {"hard", "soft"}, str, default="hard"
            Which voting method to use. Hard voting is voting based on predictions. Soft voting is voting
            based on confidences.
        conf_cols_to_est_label: dict, default=None
            A dict of dicts detailing for each predictor name (column names of base_est_predictions) the relationship
            between the class labels and the column names of confidences (column names of base_est_confidences).
        """
        self.voting = voting
        self.conf_cols_to_est_label = conf_cols_to_est_label

        # -- Validate
        if voting not in ["hard", "soft"]:
            raise ValueError("Unknown/not-supported voting method: {}".format(voting))

        if (voting == "soft") and (conf_cols_to_est_label is None):
            raise ValueError("We require details about the input to infer which column responds to which class and" +
                             " predictor. conf_cols_to_est_label can not be None!")

    def predict(self, X: pd.DataFrame, base_est_predictions: pd.DataFrame,
                base_est_confidences: Optional[pd.DataFrame] = None) -> np.ndarray:
        """

        Parameters
        ----------
        X : {array-like, pd.DataFrame}
            Test input samples
        base_est_predictions : pd.DataFrame
            Test Predictions of base models.
            Each row represent an instance and the column a base model's predictions.
        base_est_confidences: pd.DataFrame, default=None
            Test confidences of the base models for each label. Required for stack_method="predict_proba".

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predictions of the meta-estimator
        """

        if self.voting == "soft" and (base_est_confidences is None):
            raise ValueError("Require confidence values to simulate soft voting.")

        if self.voting == "hard":
            # Get majority
            preds = base_est_predictions.mode(axis=1)[0].to_numpy()
        else:
            # Preprocess Proba
            processed_proba = pd.DataFrame()
            class_labels = list(list(self.conf_cols_to_est_label.items())[0][1].keys())
            pred_cols = list(self.conf_cols_to_est_label.keys())
            # Get sum over probas for each class and store it for later argmax
            for class_label in class_labels:
                conf_cl = [self.conf_cols_to_est_label[pred_name][class_label]
                           for pred_name in pred_cols]
                processed_proba[class_label] = base_est_confidences[conf_cl].sum(axis=1)

            # Get Soft Majority
            preds = processed_proba.idxmax(axis=1).to_numpy()

        return preds


class SimulatedDynamicClassifierSelector:
    """A Simulated Dynamic Classifier Selector. Instead of aggregating prediction, this selects a prediction to use.
        Moreover, it always trains on the original X. Can speed up inference time and predictive quality.

        FIXME currently default implementation is for a regressor as meta_selector, do two classes or combine it here
            Moreover, currently 1-EPM for all example_algorithms at once. Need to abstract this to pass any selector
                own selector classes (like RF)
    """

    def __init__(self, meta_epm: BaseEstimator, stack_method: str = "predict_proba",
                 conf_cols_to_est_label: Optional[dict] = None):
        """

        Parameters
        ----------
        meta_epm : sklearn estimator/pipeline
            Empirical Performance Model (EPM) used to predict the performance of the base models. The predicted
            performance is used for dynamic selection.
        stack_method : str, default="predict_proba"
            Used stacking method. That is, what type of input is passed to the Meta-Estimator from the base models.
        conf_cols_to_est_label: dict, default=None
            A dict of dicts detailing for each predictor name (column names of base_est_predictions) the relationship
            between the class labels and the column names of confidences (column names of base_est_confidences).
        """
        self.meta_epm = meta_epm  # FIXME assume epm-regressor
        self.stack_method = stack_method
        self.conf_cols_to_est_label = conf_cols_to_est_label
        self.selected_indices = None

        # -- Validate
        if stack_method not in ["predict_proba", "predict"]:
            raise ValueError("Unknown/not-supported stack method: {}".format(stack_method))

        if (stack_method == "predict_proba") and (conf_cols_to_est_label is None):
            raise ValueError("We require details about the input to infer which column responds to which class and" +
                             " predictor. conf_cols_to_est_label can not be None!")

    def _simulate_and_build_y(self, ground_truth, base_est_predictions, base_est_confidences=None):
        if self.stack_method == "predict_proba":
            return self._preprocess_proba(base_est_predictions, base_est_confidences, ground_truth)

        return self._preprocess_predictions(base_est_predictions, ground_truth)

    def _preprocess_proba(self, pred_input, proba_input, ground_truth):
        # Get base data needed
        performance_data = self._preprocess_predictions(pred_input, ground_truth)
        correct_label_conf_dataframe = pd.DataFrame(columns=list(performance_data))
        class_labels = list(list(self.conf_cols_to_est_label.items())[0][1].keys())
        performance_cols = list(performance_data)

        # Below is the fast implementation we found to preprocess the proba accordingly
        for class_label in class_labels:
            indx_cl = ground_truth == class_label
            conf_cl = [self.conf_cols_to_est_label[pred_name][class_label]
                       for pred_name in performance_cols]
            # Select the subset of relevant confidences (subset of index and subset of columns) and append to df
            correct_label_conf_dataframe = pd.concat([correct_label_conf_dataframe,
                                                      proba_input.loc[indx_cl, conf_cl].set_axis(performance_cols,
                                                                                                 axis=1, inplace=False)
                                                      ], axis=0)

        # Sort both dfs based on index (index=originaly instance index) before subtraction
        performance_data.sort_index(inplace=True)
        correct_label_conf_dataframe.sort_index(inplace=True)

        # Compute an error for the confidence
        # just 1 - conf does not work, because that is not the lowest error, we want correct predictions,
        #   a1: 0.3 (C1), 0.3 (C2), 0.4(C3) with C3 as GT, error 1-0.4 = 0.6
        #   a2: 0.5 (C1), 0.05 (C2), 0.45 (C3) with C3 as GT, error 1-0.45 = 0.55 (lower error even tho wrong prediction)
        #  ! Do not forget relationship between confidence of classes!
        # (THIS IS LIKE ERROR NORMALIZATION THE QUESTION OF HOW TO DO THIS BEST, WE SHOULD LEARN IT...)
        #       meta-learner across tasks do learn best mapping for this?
        #       try maml for this...
        # TODO, find a better aggregator here
        #   instead of error of 1 for wrong prediction 1/n_classes to enable confidence prediction to dominate
        # performance_data = performance_data.replace(1,
        #                                             1/len())

        # Here we take the error of correct/wrong (i.e. 0/1) plus the confidence error.
        #   The error is zero if the predictor predicted the correct class and was absolute certain (100%) that it was
        #   this class.
        return pd.DataFrame(performance_data.to_numpy() + (1 - correct_label_conf_dataframe.to_numpy()),
                            columns=list(performance_data))

    @staticmethod
    def _preprocess_predictions(predictions_input, ground_truth):
        # Transform Predictions into binary (1/0) to be usable in regression label (EPMs)
        #   this is basically an error, error of 0 for correct prediction, error of 1 for bad prediction
        return predictions_input.apply(lambda x: x != ground_truth, axis=0).astype(int)

    def fit(self, X: pd.DataFrame, y: pd.Series, base_est_predictions: pd.DataFrame,
            base_est_confidences: Optional[pd.DataFrame] = None):
        """ Train the EPM

        Parameters
        ----------
        X : {array-like, pd.DataFrame}
            Training Input Samples
        y : pd.Series, array-like
            Training ground-truth
        base_est_predictions : pd.DataFrame
            Training Predictions of base models.
            Each row represent an instance and the column a base model's predictions.
        base_est_confidences: pd.DataFrame, default=None
            Confidences of the base models for each label. Required for default="predict_proba".
        """

        if self.stack_method == "predict_proba" and (base_est_confidences is None):
            raise ValueError("Require confidence values to simulate predict proba usage.")

        fit_y = self._simulate_and_build_y(y, base_est_predictions, base_est_confidences=base_est_confidences)

        self.meta_epm.fit(X, fit_y)

    def predict(self, X: pd.DataFrame, base_est_predictions: pd.DataFrame) -> np.ndarray:
        """Dynamically select models using the EPM and return the base model's predictions to use

        The EPM does not require the confidence values of the base models here as we only need to return a prediction.
        The selections assume that we want to minimize the performance as here, performance = error.

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
            The predictions of the meta-estimator
        """
        epm_predictions = self.meta_epm.predict(X)

        # We select the algorithm with the smallest error (we always want to w.l.o.g. minimize the performance)
        self.selected_indices = epm_predictions.argmin(axis=1)  # indices of example_algorithms to be selected

        # Selection to final prediction
        return base_est_predictions.to_numpy()[np.arange(len(base_est_predictions)), self.selected_indices]


class SimulatedEnsembleSelector:
    """An Implementation of Ensemble Selection

    Depending on your view point this not really a simulation but a normal implementation. For the sake of this work,
    we state it as simulation. A "normal" implementation would need to train the base models first.
    This is build on "Ensemble Selection from Libraries of Models"[1] and auto-sklearn's ensemble selection [2].

    Other: For ensemble_size = 1, this is the SBA.

    References
    ----------
    [1] R. Caruana, A. Niculescu-Mizil, G. Crew, and A. Ksikes, ‘Ensemble selection from libraries of models’,
        in Proceedings of the twenty-first international conference on Machine learning,
        New York, NY, USA, Jul. 2004, p. 18. doi: 10.1145/1015330.1015432.
    [2] M. Feurer, A. Klein, K. Eggensperger, J. Springenberg, M. Blum, and F. Hutter,
        ‘Efficient and Robust Automated Machine Learning’, in Advances in Neural Information Processing Systems, 2015,
        vol. 28.,
        code: https://github.com/automl/auto-sklearn/blob/master/autosklearn/ensembles/ensemble_selection.py

    TODO implement for raw predictions? (occurrence weighting, majority voting?)
    FIXME only works for classification for now
    """

    def __init__(self, ensemble_size: int, conf_cols_to_est_label: dict,
                 metric_to_use: Callable, maximize_metric: bool = True,
                 random_state: Union[int, np.random.RandomState, None] = None):
        """Init

        Parameters
        ----------
        ensemble_size : int
            Number of iterations of Ensemble Selection (i.e., the number of selected models part of the ensemble)
        conf_cols_to_est_label: dict
            A dict of dicts detailing for each predictor name the relationship
            between the class labels and the column names of confidences.
        metric_to_use : openml metric function
            The metric function that should be used to determine the ensemble performance
            Special format required due to OpenML's metrics.
        maximize_metric : bool, default=True
            Whether the metric computed by the metric function passed by metric_to_use is to be maximized or not.
        random_state : int, RandomState instance or None, default=None
            Controls randomness of random index selection for equally well performing ensembles
        """

        self.ensemble_size = ensemble_size
        self.metric_to_use = metric_to_use
        self.maximize_metric = maximize_metric
        self.conf_cols_to_est_label = conf_cols_to_est_label
        self.class_order = list(self.conf_cols_to_est_label[list(self.conf_cols_to_est_label.keys())[0]].keys())

        if random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(random_state, int):
            self.random_state = np.random.RandomState(random_state)
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            raise ValueError("Unknown random state object: {}".format(random_state))

    def _build_confidence_data(self, base_est_confidences):
        """Transform confidence data into array of shape (nr_base_models, nr_instances, nr_classes)
        with a specific order of classes

        """

        data = []

        for predictor, predictor_class_relationships in self.conf_cols_to_est_label.items():
            cols = [predictor_class_relationships[class_name] for class_name in self.class_order]
            data.append(base_est_confidences[cols].to_numpy())

        return np.array(data)

    def _conf_to_class_prediction(self, confidences_data):
        return np.array(self.class_order)[np.argmax(confidences_data, axis=1)]


    def fit(self, base_est_confidences: pd.DataFrame, y: pd.Series):
        """Do Ensemble Selection and calculate weights

        Basically the "slow" version of auto-sklearn with some minor changes
        https://github.com/automl/auto-sklearn/blob/master/autosklearn/ensembles/ensemble_selection.py

        Parameters
        ----------
        base_est_confidences: pd.DataFrame
            Training confidences of the base models for each label.
        y : pd.Series, array-like
            Training ground-truth
        """

        conf_data = self._build_confidence_data(base_est_confidences)

        ensemble_confidences = []
        ensemble_indices = []
        best_selector = max if self.maximize_metric else min

        for i in range(self.ensemble_size):
            new_ensemble_performances = []

            # Get performance of non-weighted average ensembles
            for j, pred_confidences in enumerate(conf_data):
                ensemble_confidences.append(pred_confidences)
                ensemble_mean_conf = np.mean(np.array(ensemble_confidences), axis=0)
                # ensemble_prediction = self._conf_to_class_prediction(ensemble_mean_conf)
                new_ensemble_performances.append(self.metric_to_use(y, None, ensemble_mean_conf, self.class_order))
                ensemble_confidences.pop()

            # Get best base model index
            all_best = np.argwhere(new_ensemble_performances == best_selector(new_ensemble_performances)).flatten()
            best_base_model_index = self.random_state.choice(all_best)
            ensemble_confidences.append(conf_data[best_base_model_index])
            ensemble_indices.append(best_base_model_index)

        # Use the ensemble to calculate the weights for base models
        self._calculate_weights(ensemble_indices)

    def _calculate_weights(self, ensemble_indices):
        """From the original autosklearn code"""
        ensemble_members = Counter(ensemble_indices).most_common()
        weights = np.zeros((len(self.conf_cols_to_est_label),), dtype=np.float64)
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights

    def predict(self, base_est_confidences: pd.DataFrame):
        """
        Parameters
        ----------
        base_est_confidences: pd.DataFrame
            Test confidences of the base models for each label.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predictions of the ensemble
        """

        conf_data = self._build_confidence_data(base_est_confidences)
        average = np.zeros_like(conf_data[0], dtype=np.float64)
        tmp_predictions = np.empty_like(conf_data[0], dtype=np.float64)

        for pred_confidences, weight in zip(conf_data, self.weights_):
            np.multiply(pred_confidences, weight, out=tmp_predictions)
            np.add(average, tmp_predictions, out=average)

        return self._conf_to_class_prediction(average)


class SimulatedDynamicEnsembleSelector:
    """A Simulated Dynamic Ensemble Selector.

        Uses an EPM to find base model competences (not any default version from DESlib, but AS like idea)
    """

    def __init__(self, meta_epm: BaseEstimator, stack_method: str = "predict_proba",
                 conf_cols_to_est_label: Optional[pd.DataFrame] = None, max_error_frac: float = 0.5):
        """Init

        Parameters
        ----------
        meta_epm : sklearn estimator/pipeline
            Empirical Performance Model (EPM) used to predict the performance of the base models. The predicted
            performance is used for dynamic selection.
        stack_method : str, default="predict_proba"
            Used stacking method. That is, what type of input is passed to the Meta-Estimator from the base models.
        conf_cols_to_est_label: dict, default=None
            A dict of dicts detailing for each predictor name (column names of base_est_predictions) the relationship
            between the class labels and the column names of confidences (column names of base_est_confidences).
        max_error_frac: float, range (0,1], default=0.5
            The fraction of the total error for an instance a dynamically selected subset of models must accumulate to
            be selected.
        """
        self.meta_epm = meta_epm  # FIXME assume epm-regressor
        self.stack_method = stack_method
        self.conf_cols_to_est_label = conf_cols_to_est_label
        self.selected_indices = None
        self.max_error_frac = max_error_frac  # Stopping criterion for adding base models

        # -- Validate
        if stack_method not in ["predict_proba", "predict"]:
            raise ValueError("Unknown/not-supported stack method: {}".format(stack_method))

        if (stack_method == "predict_proba") and (conf_cols_to_est_label is None):
            raise ValueError("We require details about the input to infer which column responds to which class and" +
                             " predictor. conf_cols_to_est_label can not be None!")

    def _simulate_and_build_y(self, ground_truth, base_est_predictions, base_est_confidences=None):
        if self.stack_method == "predict_proba":
            return self._preprocess_proba(base_est_predictions, base_est_confidences, ground_truth)

        return self._preprocess_predictions(base_est_predictions, ground_truth)

    def _preprocess_proba(self, pred_input, proba_input, ground_truth):

        # Get base data needed
        performance_data = self._preprocess_predictions(pred_input, ground_truth)
        correct_label_conf_dataframe = pd.DataFrame(columns=list(performance_data))
        class_labels = list(list(self.conf_cols_to_est_label.items())[0][1].keys())
        performance_cols = list(performance_data)

        # Below is the fast implementation we found to preprocess the proba accordingly
        for class_label in class_labels:
            indx_cl = ground_truth == class_label
            conf_cl = [self.conf_cols_to_est_label[pred_name][class_label]
                       for pred_name in performance_cols]
            # Select the subset of relevant confidences (subset of index and subset of columns) and append to df
            correct_label_conf_dataframe = pd.concat([correct_label_conf_dataframe,
                                                      proba_input.loc[indx_cl, conf_cl].set_axis(performance_cols,
                                                                                                 axis=1, inplace=False)
                                                      ], axis=0)

        # Sort both dfs based on index (index=originaly instance index) before subtraction
        performance_data.sort_index(inplace=True)
        correct_label_conf_dataframe.sort_index(inplace=True)

        # Compute an error for the confidence
        # just 1 - conf does not work, because that is not the lowest error, we want correct predictions,
        #   a1: 0.3 (C1), 0.3 (C2), 0.4(C3) with C3 as GT, error 1-0.4 = 0.6
        #   a2: 0.5 (C1), 0.05 (C2), 0.45 (C3) with C3 as GT, error 1-0.45 = 0.55 (lower error even tho wrong prediction)
        #  ! Do not forget relationship between confidence of classes!
        # (THIS IS LIKE ERROR NORMALIZATION THE QUESTION OF HOW TO DO THIS BEST, WE SHOULD LEARN IT...)
        #       meta-learner across tasks do learn best mapping for this?
        # TODO, find a better aggregator here
        #   instead of error of 1 for wrong prediction 1/n_classes to enable confidence prediction to dominate
        # performance_data = performance_data.replace(1,1/len())

        # Here we take the error of correct/wrong (i.e. 0/1) plus the confidence error.
        #   The error is zero if the predictor predicted the correct class and was absolute certain (100%) that it was
        #   this class.
        return pd.DataFrame(performance_data.to_numpy() + (1 - correct_label_conf_dataframe.to_numpy()),
                            columns=list(performance_data))

    @staticmethod
    def _preprocess_predictions(predictions_input, ground_truth):
        # Transform Predictions into binary (1/0) to be usable in regression label (EPMs)
        #   this is basically an error, error of 0 for correct prediction, error of 1 for bad prediction
        return predictions_input.apply(lambda x: x != ground_truth, axis=0).astype(int)

    def fit(self, X: pd.DataFrame, y: pd.Series, base_est_predictions: pd.DataFrame,
            base_est_confidences: Optional[pd.DataFrame] = None):
        """ Train the EPM

        Parameters
        ----------
        X : {array-like, pd.DataFrame}
            Training Input Samples
        y : pd.Series, array-like
            Training ground-truth
        base_est_predictions : pd.DataFrame
            Training Predictions of base models.
            Each row represent an instance and the column a base model's predictions.
        base_est_confidences: pd.DataFrame, default=None
            Confidences of the base models for each label. Required for default="predict_proba".
        """

        if self.stack_method == "predict_proba" and (base_est_confidences is None):
            raise ValueError("Require confidence values to simulate predict proba usage.")

        fit_y = self._simulate_and_build_y(y, base_est_predictions, base_est_confidences=base_est_confidences)

        self.meta_epm.fit(X, fit_y)

    def _aggregated_selected_predictions(self, per_instance_selections, base_est_predictions):
        # Default to Majority Voting for this example
        #   (which is "true" DES, otherwise it is sometimes called dynamic weighting)
        votes = np.where(per_instance_selections, base_est_predictions.to_numpy(), np.nan)

        # FIXME, do this better but for now this is the easiest way to get the results without re-coding the class
        preds = pd.DataFrame(votes).apply(pd.Series.value_counts, axis=1).idxmax(axis=1).to_numpy()

        return preds

    def predict(self, X: pd.DataFrame, base_est_predictions: pd.DataFrame) -> np.ndarray:
        """Dynamically select a set of models using the EPM and return the subsets models' aggregated predictions

        The EPM does not require the confidence values of the base models here as we only need to return a prediction.
        The selections assume that we want to minimize the performance as here, performance = error.
        We select the subset of models based on a fraction of the total predicted error.

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
            The predictions of the meta-estimator
        """
        epm_error_predictions = self.meta_epm.predict(X)

        # Get Models until error is max per instance # FIXME, most likely bad performing
        #   Basic Idea: select models until self.max_error_frac% of their combined error is accumulated
        def select(error_vals):
            err_sum = 0
            selected_idx = []
            sort_err = np.argsort(error_vals)
            max_error = sum(error_vals) * self.max_error_frac
            for err_idx in sort_err:

                # Add to selected set if lower than given max error
                if err_sum < max_error:
                    err_sum += error_vals[err_idx]
                    selected_idx.append(err_idx)

            # Catch empty case, DCS fallback
            if not selected_idx:
                selected_idx.append(sort_err[0])

            return np.array([i in selected_idx for i in range(len(error_vals))])

        per_instance_selections = np.apply_along_axis(select, 1, epm_error_predictions)
        return self._aggregated_selected_predictions(per_instance_selections, base_est_predictions)
