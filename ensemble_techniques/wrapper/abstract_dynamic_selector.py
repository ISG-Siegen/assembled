from abc import ABCMeta
import numpy as np
from scipy.special import softmax


class AbstractDynamicSelector(object):
    __metaclass__ = ABCMeta

    def __init__(self, ensemble_selection, ensemble_combination_method):
        self.ensemble_selection = ensemble_selection
        # Supported methods "voting", "soft_voting", "weighted_soft_voting"
        self.ensemble_combination_method = ensemble_combination_method

    def majority_vote(self):
        pass

    def _combine_predictions(self, base_models_predictions):
        """Depending the the combination method, different input is expected.

            "voting": raw predictions
            "soft_voting", "weighted_soft_voting": confidences

        We also expect self.selected_ensemble_indices_ to exist.
        """

        # The following is potentially not very efficient due to having to support selecting a variable amount of
        #   base models per instance...

        # Cast to array for easier usage
        pred_matrix = np.array(base_models_predictions)
        n_samples = len(self.selected_ensemble_indices_)
        y_pred = []

        # --- Combine
        if self.ensemble_combination_method == "voting":
            for i in range(n_samples):
                # Transform to "numbers" to make it easier to use
                class_names, pos = np.unique(pred_matrix[self.selected_ensemble_indices_[i], i],
                                             return_inverse=True)
                # Get majority
                y_pred.append(class_names[np.bincount(pos).argmax()])

        elif self.ensemble_combination_method == "soft_voting":
            for i in range(n_samples):
                # We only combine the confidences, the class with the highest confidence is selected later.
                y_pred.append(np.sum(pred_matrix[self.selected_ensemble_indices_[i], i, :], axis=0))

        else:
            # --weighted_soft_voting
            for i in range(n_samples):
                # We set the weights for each selected base model to the normalized error and for others to 0
                weights = np.zeros(pred_matrix.shape[0], dtype=float)
                curr_selection = self.selected_ensemble_indices_[i]

                # Use softmax to scale it appropriately, we use the negative because smaller is better w.r.t error
                weights[curr_selection] = softmax(-self.epm_predictions_[i][curr_selection])
                y_pred.append(np.sum(pred_matrix[:, i, :] * weights.reshape((-1, 1)), axis=0))

        return np.array(y_pred)

    def _select_prediction(self, base_models_predictions):
        return np.array(base_models_predictions)[self.selected_indices_,
                                                 np.arange(base_models_predictions[0].shape[0])]

    def _ds_predict(self, base_models_predictions):
        if not self.ensemble_selection:
            return self._select_prediction(base_models_predictions)

        return self._combine_predictions(base_models_predictions)
