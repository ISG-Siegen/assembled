from assembled.metatask import MetaTask
from assembledopenml.compatibility.openml_metrics import OpenMLAUROC

from ensemble_techniques.autosklearn.ensemble_selection import EnsembleSelection
from results.data_utils import get_default_preprocessing

mt = MetaTask()
mt.read_metatask_from_files("", -1)

technique_run_args = {"ensemble_size": 50,
                      "metric": OpenMLAUROC()}

fold_scores = mt.run_ensemble_on_all_folds(EnsembleSelection, technique_run_args, "autosklearn.EnsembleSelection",
                                           pre_fit_base_models=True, preprocessor=get_default_preprocessing(),
                                           use_validation_data_to_train_ensemble_techniques=True,
                                           meta_train_test_split_fraction=0.9,
                                           return_scores=OpenMLAUROC())
print(fold_scores)
print("Average Performance Ensemble Selection:", sum(fold_scores) / len(fold_scores))
