# Code Taken from here with adaptions to be usable:
# https://github.com/automl/auto-sklearn/blob/master/autosklearn/ensembles/ensemble_selection.py

from collections import Counter
from typing import List, Optional, Union, Callable

import numpy as np

from sklearn.utils import check_random_state
from ensemble_techniques.wrapper.abstract_ensemble import AbstractEnsemble
from ensemble_techniques.util.metrics import AbstractMetric


class EnsembleSelection(AbstractEnsemble):
    """An ensemble of selected algorithms

    Fitting an EnsembleSelection generates an ensemble from the the models
    generated during the search process. Can be further used for prediction.

    Parameters
    ----------
    base_models: List[Callable], List[sklearn estimators]
        The pool of fitted base models.
    metric: AbstractMetric
        The metric used to evaluate the models
    random_state: Optional[int | RandomState] = None
        The random_state used for ensemble selection.
        *   None - Uses numpy's default RandomState object
        *   int - Successive calls to fit will produce the same results
        *   RandomState - Truely random, each call to fit will produce
                          different results, even with the same object.
    """

    def __init__(self, base_models: List[Callable], ensemble_size: int, metric: AbstractMetric,
                 random_state: Optional[Union[int, np.random.RandomState]] = None,
                 use_best: bool = False) -> None:

        super().__init__(base_models, "predict_proba", "predict_proba")
        self.ensemble_size = ensemble_size
        self.metric = metric
        self.use_best = use_best

        # Behaviour similar to sklearn
        #   int - Deteriministic with succesive calls to fit
        #   RandomState - Successive calls to fit will produce differences
        #   None - Uses numpmys global singleton RandomState
        # https://scikit-learn.org/stable/common_pitfalls.html#controlling-randomness
        self.random_state = random_state

    def ensemble_fit(self, predictions: List[np.ndarray], labels: np.ndarray) -> AbstractEnsemble:
        self.ensemble_size = int(self.ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError('Ensemble size cannot be less than one!')
        if not isinstance(self.metric, AbstractMetric):
            raise ValueError("The provided metric must be an instance of a AbstractMetric, "
                             "nevertheless it is {}({})".format(
                self.metric,
                type(self.metric),
            ))

        self._fit(predictions, labels)
        self.apply_use_best()

        self._calculate_weights()

        return self

    def _fit(self, predictions: List[np.ndarray], labels: np.ndarray) -> None:
        """Fast version of Rich Caruana's ensemble selection method."""
        self.num_input_models_ = len(predictions)
        rand = check_random_state(self.random_state)

        ensemble = []  # type: List[np.ndarray]
        trajectory = []
        order = []

        ensemble_size = self.ensemble_size

        weighted_ensemble_prediction = np.zeros(
            predictions[0].shape,
            dtype=np.float64,
        )
        fant_ensemble_prediction = np.zeros(
            weighted_ensemble_prediction.shape,
            dtype=np.float64,
        )
        for i in range(ensemble_size):
            losses = np.zeros(
                (len(predictions)),
                dtype=np.float64,
            )
            s = len(ensemble)
            if s > 0:
                np.add(
                    weighted_ensemble_prediction,
                    ensemble[-1],
                    out=weighted_ensemble_prediction,
                )

            # Memory-efficient averaging!
            for j, pred in enumerate(predictions):
                # fant_ensemble_prediction is the prediction of the current ensemble
                # and should be ([predictions[selected_prev_iterations] + predictions[j])/(s+1)
                # We overwrite the contents of fant_ensemble_prediction
                # directly with weighted_ensemble_prediction + new_prediction and then scale for avg
                np.add(
                    weighted_ensemble_prediction,
                    pred,
                    out=fant_ensemble_prediction
                )
                np.multiply(
                    fant_ensemble_prediction,
                    (1. / float(s + 1)),
                    out=fant_ensemble_prediction
                )

                losses[j] = self.metric(labels, fant_ensemble_prediction, to_loss=True)

            all_best = np.argwhere(losses == np.nanmin(losses)).flatten()

            best = rand.choice(all_best)

            ensemble.append(predictions[best])
            trajectory.append(losses[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        self.indices_ = order
        self.trajectory_ = trajectory
        self.train_loss_ = trajectory[-1]

    def apply_use_best(self):
        if self.use_best:
            # Basically from autogluon the code
            min_score = np.min(self.trajectory_)
            idx_best = self.trajectory_.index(min_score)
            self.indices_ = self.indices_[:idx_best + 1]
            self.trajectory_ = self.trajectory_[:idx_best + 1]
            self.ensemble_size = idx_best + 1
            self.train_loss_ = self.trajectory_[idx_best]
        else:
            self.train_score_ = self.trajectory_[-1]

    def _calculate_weights(self) -> None:
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros(
            (self.num_input_models_,),
            dtype=np.float64,
        )
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights

    def ensemble_predict(self, predictions: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:

        average = np.zeros_like(predictions[0], dtype=np.float64)
        tmp_predictions = np.empty_like(predictions[0], dtype=np.float64)

        # if predictions.shape[0] == len(self.weights_),
        # predictions include those of zero-weight models.
        if len(predictions) == len(self.weights_):
            for pred, weight in zip(predictions, self.weights_):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)

        # if prediction model.shape[0] == len(non_null_weights),
        # predictions do not include those of zero-weight models.
        elif len(predictions) == np.count_nonzero(self.weights_):
            non_null_weights = [w for w in self.weights_ if w > 0]
            for pred, weight in zip(predictions, non_null_weights):
                np.multiply(pred, weight, out=tmp_predictions)
                np.add(average, tmp_predictions, out=average)

        # If none of the above applies, then something must have gone wrong.
        else:
            raise ValueError("The dimensions of ensemble predictions"
                             " and ensemble weights do not match!")
        del tmp_predictions
        return average

    @property
    def _to_save_metadata(self):
        return dict(
            weights=self.weights_.tolist(),
            ensemble_size=int(sum(self.weights_ > 0)),
            train_score=float(self.train_score_),
            trajectory=[float(v) for v in self.trajectory_])
