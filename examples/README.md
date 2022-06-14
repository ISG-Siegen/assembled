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
