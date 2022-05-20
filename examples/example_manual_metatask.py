"""Example for building a metatask from your own data without OpenML

"""
import numpy as np

from assembled.metatask import MetaTask

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.datasets import load_breast_cancer  # load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# Select dataset
sklearn_load_dataset_function = load_breast_cancer
metatask_id = -1  # Negative number because not an OpenML task!

# Load a Dataset as Dataframe
task_data = sklearn_load_dataset_function(as_frame=True)
target_name = task_data.target.name
dataset_frame = task_data.frame
class_labels = np.array([str(x) for x in task_data.target_names])  # cast labels to string
feature_names = task_data.feature_names
cat_feature_names = []
# Make target column equal class labels again (inverse of label encoder)
dataset_frame[target_name] = class_labels[dataset_frame[target_name].to_numpy()]

# Start Filling Metatask with task data
mt = MetaTask()

# Get Splits
fold_indicators = np.empty(len(dataset_frame))
cv_spliter = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
for fold_idx, (train_index, test_index) in enumerate(
        cv_spliter.split(dataset_frame[feature_names], dataset_frame[target_name])):
    # This indicates the test subset for fold with number fold_idx
    fold_indicators[test_index] = fold_idx

mt.init_dataset_information(dataset_frame, target_name=target_name, class_labels=class_labels,
                            feature_names=feature_names, cat_feature_names=cat_feature_names,
                            task_type="classification", openml_task_id=metatask_id,
                            dataset_name="breast_cancer", folds_indicator=fold_indicators
                            )

# Get Cross-val-predictions for some base models
base_models = [
    ("RF_10", RandomForestClassifier(n_estimators=10)),
    ("RF_100", RandomForestClassifier(n_estimators=100)),
    ("RF_200", RandomForestClassifier(n_estimators=200)),
    ("GB_10", HistGradientBoostingClassifier(loss="auto", max_iter=10)),
    ("GB_100", HistGradientBoostingClassifier(loss="auto", max_iter=100)),
]

for bm_name, bm in base_models:
    classes = np.unique(dataset_frame[target_name])  # because cross_val_predict does not return .classes_
    bm_confidence = cross_val_predict(bm, dataset_frame[feature_names], dataset_frame[target_name], cv=cv_spliter,
                                      n_jobs=2, method="predict_proba")
    bm_predictions = classes.take(np.argmax(bm_confidence, axis=1), axis=0)  # Get original class names instead int

    mt.add_predictor(bm_name, bm_predictions, confidences=bm_confidence, conf_class_labels=list(classes),
                     predictor_description=str(bm))

# Add Selection Constraints
mt.read_selection_constraints({"manual": True})

# Save Metatask
mt.to_files(output_dir="../results/metatasks")
