# Assembled-OpenML for an Algorithm Selection Use Case (Test Version) 

Assembled-OpenML is a framework to build Meta-Datasets (called Metatasks) from OpenML. In this version,
the Metatasks contain the predictions and confidences (e.g. sklearn's predict_proba) of specific OpenML runs and 
the data of the original task. The predictions that shall be stored within a metatasks can be filtered based on several 
settings. Here, the predictions correspond to the best run (configuration) of an OpenML FLow (ML algorithm/pipeline).


## A Comment on Base Model Diversity for this Version

The automatically selected algorithms are not necessarily divers. While we filter version duplicates, we can not filter
duplicates based on descriptions. For example, two different flow might both use Random Forest as classifier but
different preprocessing steps. While we could filter some duplicated algorithms based on the description, this becomes
problematic across libraries. For example, the random forest implementation of mlr3 is called "ranger" and of sklearn "
RandomForestClassifier". Manually effort (e.g. removing duplicates manually)
or data from OpenML Uploader ML Libraries can fix this.

* Filtering the Metatasks and their predictors for a benchmark using crawled Metatasks (like in
  the `benchmark_set_search.py` as shown in other branches) can also alleviate this.

## Default Crawler Settings

* We only look at Tasks tagged with "OpenML-CC18".
* We search for the best performing flows (and their runs) based on the "area under roc curve" metric.
* Each flow must have at least 10 runs.
* No Version Duplicates allowed
* We take at most the 50 top-performing flows for a task.
