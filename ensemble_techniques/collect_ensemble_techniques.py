# -- Imports for Ensemble Techniques
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from deslib.dcs import APosteriori, APriori, LCA, MCB, MLA, OLA, Rank
from deslib.des import METADES, DESClustering, DESP, DESKNN, KNOP, KNORAE, KNORAU
from deslib.des.probabilistic import RRC, DESKL, MinimumDifference, Logarithmic, Exponential
from ensemble_techniques.autosklearn.ensemble_selection import EnsembleSelection
from ensemble_techniques.custom.baselines import SingleBest, VirtualBest
from ensemble_techniques.custom.ds_epm import DSEmpiricalPerformanceModel
from ensemble_techniques.sklearn.stacking import StackingClassifier
from ensemble_techniques.sklearn.voting import VotingClassifier

# -- Get Preprocessing
from results.data_utils import get_default_preprocessing

# -- Get Metric
from assembledopenml.compatibility.openml_metrics import OpenMLAUROC

# -- Randomness 
from numpy.random import RandomState


def get_sklearn_techniques(rng_seed):
    # -- Sklearn Existing Models
    # Could use existing methods but they break with calibration + fake models (due to datasets being too small for cv)
    sklearn_techniques = {
        "sklearn.VotingClassifier": {
            "technique": VotingClassifier,
            "technique_args": {"voting": "soft",
                               # , "n_jobs": -1
                               "prefitted": True,
                               },
            "probability_calibration": "auto",
            "pre_fit_base_models": True
        },
        "sklearn.StackingClassifier": {
            "technique": StackingClassifier,
            "technique_args": {"final_estimator": LogisticRegression(random_state=RandomState(rng_seed), n_jobs=-1),
                               # "n_jobs": -1
                               "prefitted": True, "blending": True
                               },
            "probability_calibration": "auto",
            "pre_fit_base_models": True
        },
        "sklearn.StackingClassifier.passthrough": {
            "technique": StackingClassifier,
            "technique_args": {"final_estimator": RandomForestClassifier(random_state=RandomState(rng_seed), n_jobs=-1),
                               # "n_jobs": -1
                               "prefitted": True, "blending": True
                               },
            "probability_calibration": "auto",
            "pre_fit_base_models": True
        },
    }
    # Add default values for sklearn
    sklearn_default_values = {"base_models_with_names": True, "label_encoder": True,
                              "preprocessor": get_default_preprocessing()}
    for key in sklearn_techniques.keys():
        sklearn_techniques[key].update(sklearn_default_values)

    return sklearn_techniques


def get_deslib_techniques(rng_seed):
    # -- DESLib Existing Models
    deslib_techniques = {
        # DCS Methods
        "DCS.APosteriori": {"technique": APosteriori,
                            "technique_args": {"random_state": RandomState(rng_seed), "n_jobs": -1},
                            "probability_calibration": "auto"},
        "DCS.APriori": {"technique": APriori,
                        "technique_args": {"random_state": RandomState(rng_seed), "n_jobs": -1},
                        "probability_calibration": "auto"},
        "DCS.LCA": {"technique": LCA,
                    "technique_args": {"random_state": RandomState(rng_seed), "n_jobs": -1}},
        "DCS.MCB": {"technique": MCB,
                    "technique_args": {"random_state": RandomState(rng_seed), "n_jobs": -1}},
        "DCS.MLA": {"technique": MLA,
                    "technique_args": {"random_state": RandomState(rng_seed), "n_jobs": -1}},
        "DCS.OLA": {"technique": OLA,
                    "technique_args": {"random_state": RandomState(rng_seed), "n_jobs": -1}},
        "DCS.Rank": {"technique": Rank,
                     "technique_args": {"random_state": RandomState(rng_seed), "n_jobs": -1}},

        # DES Methods
        "DES.METADES": {"technique": METADES,
                        "technique_args": {"random_state": RandomState(rng_seed), "n_jobs": -1}},
        "DES.DESClustering": {"technique": DESClustering,
                              "technique_args": {"random_state": RandomState(rng_seed), "n_jobs": -1,
                                                 # Switch to accuracy instead of original score metric,
                                                 #   because implementation does not support AUROC correctly.
                                                 "metric_performance": "accuracy_score",

                                                 # have to init by us because it is bugged otherwise
                                                 #   (most likely wrong python/sklearn version used by us)
                                                 #   but kmeans never had n_jobs?...
                                                 "clustering": KMeans(n_clusters=5, random_state=RandomState(rng_seed))
                                                 }
                              },
        "DES.DESP": {"technique": DESP,
                     "technique_args": {"random_state": RandomState(rng_seed), "n_jobs": -1}},
        "DES.DESKNN": {"technique": DESKNN,
                       "technique_args": {"random_state": RandomState(rng_seed), "n_jobs": -1}},
        "DES.KNOP": {"technique": KNOP,
                     "technique_args": {"random_state": RandomState(rng_seed), "n_jobs": -1}},
        "DES.KNORAE": {"technique": KNORAE,
                       "technique_args": {"random_state": RandomState(rng_seed), "n_jobs": -1}},
        "DES.KNORAU": {"technique": KNORAU,
                       "technique_args": {"random_state": RandomState(rng_seed), "n_jobs": -1}},

        # Probabilistic
        "Probabilistic.RRC": {"technique": RRC,
                              "technique_args": {"random_state": RandomState(rng_seed), "n_jobs": -1}},
        "Probabilistic.DESKL": {"technique": DESKL,
                                "technique_args": {"random_state": RandomState(rng_seed), "n_jobs": -1}},
        "Probabilistic.MinimumDifference": {"technique": MinimumDifference,
                                            "technique_args": {"random_state": RandomState(rng_seed), "n_jobs": -1}},
        "Probabilistic.Logarithmic": {"technique": Logarithmic,
                                      "technique_args": {"random_state": RandomState(rng_seed), "n_jobs": -1}},
        "Probabilistic.Exponential": {"technique": Exponential,
                                      "technique_args": {"random_state": RandomState(rng_seed), "n_jobs": -1}},

        # Static
        #   We are not using any of the static method from deslib.
        #   The SBA and VBA are suboptimal and we already have stacking from sklearn.

    }
    # Add default values for sklearn
    deslib_default_values = {"pre_fit_base_models": True, "preprocessor": get_default_preprocessing()}
    for key in deslib_techniques.keys():
        deslib_techniques[key].update(deslib_default_values)
    deslib_techniques = {"deslib.{}".format(key): val for key, val in deslib_techniques.items()}

    return deslib_techniques


def get_autosklearn_techniques(rng_seed):
    autosklearn_techniques = {
        "autosklearn.EnsembleSelection": {
            "technique": EnsembleSelection,
            "technique_args": {"ensemble_size": 50,
                               "metric": OpenMLAUROC(),  # Please be aware, AUROC is very inefficient
                               "bagging": False, "mode": "fast",
                               "random_state": RandomState(rng_seed)},
            "probability_calibration": "auto"
        }
    }

    autosklearn_default_values = {"pre_fit_base_models": True, "preprocessor": get_default_preprocessing()}
    for key in autosklearn_techniques.keys():
        autosklearn_techniques[key].update(autosklearn_default_values)

    return autosklearn_techniques


def get_custom_techniques(rng_seed):
    new_techniques = {
        "custom.DSEmpiricalPerformanceModel.DES.wsv": {
            "technique": DSEmpiricalPerformanceModel,
            "technique_args": {"epm": RandomForestRegressor(random_state=RandomState(rng_seed), n_jobs=-1),
                               "epm_error": "predict_proba",
                               "ensemble_selection": True, "ensemble_combination_method": "weighted_soft_voting"},
            "probability_calibration": "auto"
        },
        "custom.DSEmpiricalPerformanceModel.DCS": {
            "technique": DSEmpiricalPerformanceModel,
            "technique_args": {"epm": RandomForestRegressor(random_state=RandomState(rng_seed), n_jobs=-1),
                               "epm_error": "predict_proba"},
            "probability_calibration": "auto"
        },
    }
    baselines = {
        "custom.VirtualBest": {
            "technique": VirtualBest,
            "technique_args": {"handle_no_correct": True},
            "oracle": True,
            "probability_calibration": "auto"
        },
        "custom.SingleBest": {
            "technique": SingleBest,
            "technique_args": {"metric": OpenMLAUROC(),  # Please be aware, AUROC is very inefficient
                               "predict_method": "predict_proba"},
            "probability_calibration": "auto"
        }
    }
    custom_techniques = {**new_techniques, **baselines}
    custom_default_values = {"pre_fit_base_models": True, "preprocessor": get_default_preprocessing()}
    for key in custom_techniques.keys():
        custom_techniques[key].update(custom_default_values)

    return custom_techniques


def get_benchmark_techniques(rng_seed):
    return {**get_sklearn_techniques(rng_seed), **get_deslib_techniques(rng_seed),
            **get_autosklearn_techniques(rng_seed), **get_custom_techniques(rng_seed)}
