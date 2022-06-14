# Examples of using Assembled and Assembled-OpenML

## Examples and Short Descriptions

### Manual Data

* `example_manual_metatask.py`: An example of how to build a metatask from your own data. This includes filling a
  metatask with dataset and task information. Moreover, it is shown how to add the predictions collected from (sklearn)
  base models to a metatask.
* `example_manual_metatask_with_validation_data.py`: An example of how to build a metatask from your own data and from
  OpenML. That is, filling a metatask with dataset and task information from OpenML. While filling the predictor
  information on your own. This allows you to add additional validation data (which base models on OpenML do not have).
  In this example, we also show how to evaluate with validation data.
* `example_use_benchmaker.py.py`: Example code that shows the usage of the BenchMaker, the part of Assembled that allows
  us to build and more easily share/reproduce metatasks. The code first creates a benchmark from a set of metatask and
  saves information about the benchmark in a `benchmark_details.json`. In this example, it also creates a .zip file used
  to share the metadata (including prediction data) of a metatask. Next, it rebuilds the benchmark's metatask from
  the `benchmark_details.json` and .zip. Requires `example_manual_metatask` to have been run before.

### OpenML Data

* `example_openml_cc18.py`: An example on how to use the task IDs from the curated classification benchmark OpenML-CC18
  to build and save metatasks. As implied by one of the comments, you could also pass a list of task IDs by hand.
* `example_rebuild_openml_benchmark.py`: The code that can be used to re-build a benchmark from
  a `benchmark_details.json` created by the BenchMaker from Assembled for OpenML tasks. This allows the user to get the
  exact same benchmark metatasks as created by another user.

### General

* `example_run_benchmark.py`: Code that shows how to run a benchmark using metatasks and ensemble techniques. It uses
  the functionalities of a metataks to perform an evaluation. Moreover, it uses ensemble techniques (as initialized
  in `/ensemble_techniques/`). By default, the code will save the output (ensemble techniques' predictions) in a
  directory. The ensemble techniques evaluated with the metatasks are run by simulating the base model through our faked
  predictors (e.g., FakedClassifier) classes. This evaluation is without validation data by default.
* `example_evaluation_and_analysis.py` The code runs an evaluation and analysis of the output of a benchmark (as created
  by the metatask evaluation functionality). Currently, it produces plots and data to determine the average best
  ensemble technique.

## Default Example Benchmark Settings

### OpenMLAssembler Settings

* We only look at Tasks tagged with "OpenML-CC18".
* We search for the 50 performing unique configurations (runs) based on the "area under roc curve" metric.

### Post-Processing Settings - How we filter Metatasks and their Predictors

Our benchmark set search script has several parameters to control our search procedure. We set the parameters to do the
following:

* The GAP between the performance of the VBA and SBA must be at least 5%.
* The performance is based on OpenML's area_under_roc_curve
* We require at least 10 base models to represent a valid metatask.
* We filter bad base models (e.g. base models with corrupted and non-fixable prediction data). Bad base models have been
  flagged during the creation of a metatask.
* We filtered worse than random base models.

### Resulting Task IDs and Predictors

Please see `results/benchmark_metatasks/benchmark_details.json` for the full list of valid task IDs and valid predictors
per task. This benchmark can be re-build with the exact same data (see `example_rebuild_benchmark_data.py` for details).

# Known Issues

## Confidence Values (Equal, ...)

Assembled-OpenML collects data from OpenML. This data includes confidence values of base models for classes. These
confidence values can be equal across all classes for multiple reasons (we made it equal to fix a bug; they were equal
from the start; ...). Moreover, our code guarantees for collected data, that the confidence value of the predicted class
is equal to the highest confidences. If the code can not guarantee this, the base model is marked in the .json file.

This is very much relevant for anyone working with the prediction data. If you were to naively take the argmax of the
confidences to get the prediction, the resulting prediction could be unequal to the prediction of the base model if
multiple classes have the same highest confidence. For evaluation or other use-cases (like oracle-predictors), one needs
to be careful with the argmax of the confidences to not run into problems resulting from a (wrong) difference between
the argmax of the prediction and the actual prediction of a base model.

## Executing Examples or Experiments in the CLI

If you execute python scripts from the CLI (and not from an IDE) outside the root directory, the imports might not work.
To avoid this, add the root directory to the PYTHONPATH (this is just one possible solution).

For example, to execute the `example_rebuild_benchmark_data.py` while being in the `examples/openml_data`
directory:

```bash
PYTHONPATH=../../ python example_rebuild_openml_benchmark.py
```

## Problems with Implementations of Ensemble Techniques

Please see the README of the `ensemble_techniques` for more details on problems with the implementations of used
ensemble techniques.
