from numpy.random import RandomState
from assembledopenml.metatask import MetaTask
from experiments.data_utils import get_valid_benchmark_ids

# -- Imports for Ensemble Techniques
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from deslib.dcs import APosteriori, APriori, LCA, MCB, MLA, OLA, Rank
from deslib.des import METADES, DESClustering, DESP, DESKNN, KNOP, KNORAE, KNORAU
from deslib.des.probabilistic import RRC, DESKL, MinimumDifference, Logarithmic, Exponential


# -- New/Better Preprocessing
def get_default_preprocessing():
    from sklearn.compose import ColumnTransformer
    from sklearn.compose import make_column_selector
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="constant", fill_value=-1),
             make_column_selector(dtype_exclude="category")),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"),
             make_column_selector(dtype_include="category")),
        ],
        sparse_threshold=0
    )

    return preprocessor


# -- Init Ensemble Techniques
def get_benchmark_techniques(random_state_object):
    # -- Sklearn Existing Models
    sklearn_techniques = {
        "sklearn.VotingClassifier": {
            "technique": VotingClassifier,
            "technique_args": {"voting": "hard", "n_jobs": -1},
            "fit_technique_on_original_data": True
        },
        # Stacking but not trained on cross-val predictions but on default meta-train split achieved by fake base models
        "sklearn.StackingClassifier": {
            "technique": StackingClassifier,
            "technique_args": {"final_estimator": LogisticRegression(random_state=random_state_object, n_jobs=-1),
                               # "n_jobs": -1 # FIXME: potential bug that n_jobs -1 does not work on windows
                               },
        }
    }
    # Add default values for sklearn
    sklearn_default_values = {"pre_fit_base_models": False, "base_models_with_names": True, "label_encoder": True,
                              "preprocessor": get_default_preprocessing()}
    for key in sklearn_techniques.keys():
        sklearn_techniques[key].update(sklearn_default_values)

    # -- DESLib Existing Models
    deslib_techniques = {
        # DCS Methods
        "DCS.APosteriori": {"technique": APosteriori,
                            "technique_args": {"random_state": random_state_object}},
        "DCS.APriori": {"technique": APriori,
                        "technique_args": {"random_state": random_state_object}},
        "DCS.LCA": {"technique": LCA,
                    "technique_args": {"random_state": random_state_object}},
        "DCS.MCB": {"technique": MCB,
                    "technique_args": {"random_state": random_state_object}},
        "DCS.MLA": {"technique": MLA,
                    "technique_args": {"random_state": random_state_object}},
        "DCS.OLA": {"technique": OLA,
                    "technique_args": {"random_state": random_state_object}},
        "DCS.Rank": {"technique": Rank,
                     "technique_args": {"random_state": random_state_object}},

        # DES Methods
        "DES.METADES": {"technique": METADES,
                        "technique_args": {"random_state": random_state_object}},
        "DES.DESClustering": {"technique": DESClustering,
                              "technique_args": {"random_state": random_state_object,
                                                 # Switch to accuracy instead of original score metric,
                                                 #   because implementation does not support AUROC correctly.
                                                 "metric_performance": "accuracy_score",

                                                 # have to init by us because it is bugged otherwise
                                                 #   (most likely wrong python/sklearn version used by us)
                                                 #   but kmeans never had n_jobs?...
                                                 "clustering": KMeans(n_clusters=5, random_state=rng)
                                                 }
                              },
        "DES.DESP": {"technique": DESP,
                     "technique_args": {"random_state": random_state_object}},
        "DES.DESKNN": {"technique": DESKNN,
                       "technique_args": {"random_state": random_state_object}},
        "DES.KNOP": {"technique": KNOP,
                     "technique_args": {"random_state": random_state_object}},
        "DES.KNORAE": {"technique": KNORAE,
                       "technique_args": {"random_state": random_state_object}},
        "DES.KNORAU": {"technique": KNORAU,
                       "technique_args": {"random_state": random_state_object}},

        # Probabilistic
        "Probabilistic.RRC": {"technique": RRC,
                              "technique_args": {"random_state": random_state_object}},
        "Probabilistic.DESKL": {"technique": DESKL,
                                "technique_args": {"random_state": random_state_object}},
        "Probabilistic.MinimumDifference": {"technique": MinimumDifference,
                                            "technique_args": {"random_state": random_state_object}},
        "Probabilistic.Logarithmic": {"technique": Logarithmic,
                                      "technique_args": {"random_state": random_state_object}},
        "Probabilistic.Exponential": {"technique": Exponential,
                                      "technique_args": {"random_state": random_state_object}},

        # Static
        #   We are not using any of the static method from deslib.
        #   The SBA and VBA are suboptimal and we already have stacking from sklearn.

    }
    # Add default values for sklearn
    deslib_default_values = {"pre_fit_base_models": True, "base_models_with_names": False, "label_encoder": False,
                             "preprocessor": get_default_preprocessing()}
    for key in deslib_techniques.keys():
        deslib_techniques[key].update(deslib_default_values)
    deslib_techniques = {"deslib.{}".format(key): val for key, val in deslib_techniques.items()}

    # -- Custom Methods

    return {**sklearn_techniques, **deslib_techniques}


# -- Main
if __name__ == "__main__":

    # --- Input para
    valid_task_ids = get_valid_benchmark_ids()[:2]
    test_split_frac = 0.5
    rng = RandomState(3151278530)
    # The following is not a random state object nor part of the above rng to avoid the problem that adding
    #   a new technique or dataset would change the random state and thus make it less comparable across runs.
    test_split_rng = 581640921
    techniques_to_benchmark = get_benchmark_techniques(rng)

    # --- Iterate over tasks to gather results
    nr_tasks = len(valid_task_ids)
    for task_nr, task_id in enumerate(valid_task_ids, start=1):
        mt = MetaTask()
        mt.read_metatask_from_files("../results/benchmark_metatasks", task_id)
        print("#### Process Task {} for dataset {} ({}/{}) ####".format(mt.openml_task_id, mt.dataset_name,
                                                                        task_nr, nr_tasks))
        out_path = "../results/new_benchmark_output/results_for_metatask_{}.csv".format(task_id)

        # -- Run Techniques on Metatask
        nr_techniques = len(techniques_to_benchmark)
        counter_techniques = 1
        for technique_name, technique_run_args in techniques_to_benchmark.items():
            print("### Benchmark Ensemble Technique: {} ({}/{})###".format(technique_name, counter_techniques,
                                                                           nr_techniques))
            counter_techniques += 1
            mt.run_ensemble_on_all_folds(technique_name=technique_name, **technique_run_args,
                                         meta_train_test_split_fraction=0.5, output_file_path=out_path,
                                         meta_train_test_split_random_state=test_split_rng)

    # ----------------- Problems
    # 1) can not do multiprocessing for sklearn stacking on windows due to a bug
    # 2) can not pass the openml roc auc metric as deslib requires the string
    # 3) can not pass roc_auc_score to DESLib
    # 4) DESLIb's oracle does not work and is "primitive" to some extent in relation to confidences

    # ----------------- TODOs
    # 1) add custom functions methods (for sba, vba)?
    # 2) add autosklearn methods?
