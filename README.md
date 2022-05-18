# Assembled

Assembled is planed to be a framework for ensemble evaluation. It shall be able to run, benchmark, and evaluate ensemble
techniques without the overhead of training base models.

Currently, its main features are:

* **Metatasks**: A metatasks is a meta-dataset and a class interface. Metatasks contain the predictions and
  confidences (e.g. sklearn's predict_proba) of specific base models and the data of an original (OpenML) task.
  Moreover, its class interface contains several useful method to simplify the evaluation and benchmarking of ensemble
  techniques. A collection of metatasks can be used to benchmark ensemble techniques without the computational overhead
  of training and evaluating base models.
* **Assembled-OpenML**: an extension of Assembled to build metatasks with data from OpenML (its original design). Only a
  OpenML Task ID must be passed to the code to generate a metatask for a specific OpenML task.
  Technically, any ID can be passed. In practice, only supervised classification tasks are supported so far.
  Moreover, this tool was build for and tested against curated benchmarks (like tasks in OpenMLCC-18). 
  Other classification tasks should be supported as well but bugs might be more likely. 
* **FakedClassifiers**: Code to simulate the behavior of a base model by passing it appropriate data during the
  initialization. Allows us to evaluate most ensemble techniques without code changes to the original implementation's
  code.
* **Supporting Ensemble Techniques** We created code to make ensemble techniques usable with (pre-fitted) base models.
  This is not part of Assembled itself but rather additional example on how to use ensemble techniques with Assembled. 
  Some implementation support base models by default other do not. See `/ensemble_techniques/` for more details.

Currently, is main use-cases are:

* Ensembles After AutoML (Post-Processing)

This repository/branch also contains the Assembled-OpenML extension.

## Assembled-OpenML

_For the original code of the workshop paper on Assembled-OpenML, see the `automl_workshop_paper` branch_

Assembled-OpenML builds Metatasks from OpenML. In this first version of Assembled-OpenML, the predictions correspond to
the top-n best runs (configurations) of an OpenML task. It shall simulate the use case of post-processing an AutoML
tool's top-n set of configurations.

Assembled-OpenML enables the user to quickly generate a benchmark set by entering a list of OpenML Task IDs as input
(see our code examples). In general, Assembled-OpenML is an affordable/efficient alternative to creating benchmarks by
hand. It is affordable/efficient, because you do not need to train and evaluate the base models but can directly
evaluate ensemble techniques.

# For the User

## Installation

The code in this repository was tested and developed for Python Version 3.9 on Linux and Windows. To install the
requirements for all code stored in this repository, please set up a python environment with our `requirements.txt`. If
needed, install Python or create a virtual environment.

In the environment execute from the project root:

```bash
pip install -r requirements.txt
```

Please be aware that any relevant-enough subdirectory keeps track of its own requirements through a `requirements.txt`.
Hence, if you want to use only parts of this project, it might be a better idea to only install the requirements of the
code that you want to use.

## Usage

To see the example usage of Assembled-OpenML, see the `./examples/` directory for code examples and more details.

A simple example of using Assembled-OpenML is:

```python
from assembledopenml.openml_assembler import OpenMLAssembler

omla = OpenMLAssembler(nr_base_models=50)
meta_task = omla.run(openml_task_id=3)  # Build a metatask for the OpenML task with ID 3 
meta_task.to_files("results/metatasks")  # Store the meta-dataset 
```

A simple example of evaluating auto-sklearn's ensemble selection with a Metatask is:

```python
from assembled.metatask import MetaTask
from assembledopenml.compatibility.openml_metrics import OpenMLAUROC
from numpy.random import RandomState
from results.data_utils import get_default_preprocessing

# Import an adapted version of auto-sklearn's Ensemble Selection 
# (basically a wrapper since the original implementation only works with predictions and not base models)
from ensemble_techniques.autosklearn.ensemble_selection import EnsembleSelection

mt = MetaTask()
# We assume here, that the metatask files for task ID 3 exist. Alternatively, call omla.run(openml_task_id=3) first.
mt.read_metatask_from_files("results/metatasks", 3)

# Define the ensemble technique (arguments)
technique_run_args = {"ensemble_size": 50,
                      "metric": OpenMLAUROC(),  # Please be aware, AUROC is very inefficient
                      "random_state": RandomState(0)
                      }

fold_scores = mt.run_ensemble_on_all_folds(EnsembleSelection, technique_run_args, "autosklearn.EnsembleSelection",
                                           pre_fit_base_models=True, preprocessor=get_default_preprocessing(),
                                           meta_train_test_split_fraction=0.5, meta_train_test_split_random_state=0,
                                           return_scores=OpenMLAUROC())
print(fold_scores)
print("Average Performance:", sum(fold_scores) / len(fold_scores))
```

To re-use our evaluation code, coined Assembled-Evaler for now, a benchmark must have been run first,
see `example_run_benchmark.py` and `example_rebuild_benchmark_data.py`. Afterwards,
see `example_evaluation_and_analysis.py` to run our pre-defined evaluation code.

## Limitations

* **Regression is not supported** so far as OpenML has not enough data (runs) on Regression tasks. Would require some
  additional implementations.
* Assembled-OpenML ignores OpenMl repetitions (most runs/datasets do not provide repetitions).
* The file format for the predictions file is not fully standardized in OpenML and hence requires manually adjustment to
  all used formats. Hopefully, we found most of the relevant formats with Assembled-OpenML.
* Some files, which store predictions, seem to have malicious or corrupted predictions/confidence values. If we can not
  fix such a case, we store these bad predictors in the Metatask object to be manually validated later on. Moreover,
  these predictors can be filtered from the Metatask if one wants to (we do this for every example or experiment).

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

### Benchmark Run and Supported Ensemble Techniques

All details on already supported ensemble techniques and libraries (like sklearn or DESlib) can be found in the
`ensemble_techniques` directory.

It is important to note, that we utilize ensemble techniques of existing libraries and new/custom techniques through
simulating base models based on our data. To execute ensemble techniques on our benchmark, we have created so-called "
FakedClassifiers". These represent base models (sklearn estimators) to a library like sklearn or DESlib and thus allows
us to use these natively. Employing these fake base models is the default case for our benchmark and all our code. We
have also added a wrapper to use methods that only work with the predictions (like autosklearn's ensemble selection).

The fake base models are created on-the-fly when you run the "run_ensemble_on_all_folds" function of a Metatask and then
passed to the ensemble technique (we assume that a list of base models or something similar is the first argument of the
technique's initialization).

Please consult the first version of Assembled-OpenML (see the other branches) to see an alternative on how to use/create
a benchmark of ensemble techniques that works with the data directly instead of base models.

# For Developers

This section is very much work in progress...

## Unit Tests

We use [py.test](https://docs.pytest.org/en/latest/) for unit testing.

After installing it, while in the root directory (and in the current environment), call:

```bash
python -m pytest tests/
```

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
To avoid this, add the root directory to the PYTHONPATH (one possible solution).

For example, to time the execution of the `example_rebuild_benchmark_data.py` while being in the `examples` directory:

```bash
time PYTHONPATH=~/path-to/assembled python example_rebuild_benchmark_data.py 
```

## Problems with Implementations of Ensemble Techniques

Please see the README of the `ensemble_techniques` for more details on problems with the implementations of used
ensemble techniques. 