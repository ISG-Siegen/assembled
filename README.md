# assembled

Assembled is planed to be a framework for ensemble evaluation. It shall run, benchmark, and evaluate ensemble techniques
without the overhead of training base models.

Currently, its main features are:

* **Metatasks**: A metatasks is a meta-dataset and a class interface. Metatasks contain the predictions and
  confidences (e.g. sklearn's predict_proba) of specific base models and the data of an original (OpenML) task.
  Moreover, its class interface contains several useful method to simplify the evaluation and benchmarking of ensemble
  techniques. A collection of metatasks can be used to benchmark ensemble techniques without the computational overhead
  of training and evaluating base models.
* **Assembled-OpenML**: an extension of Assembled to build metatasks with data from OpenML. Only a OpenML Task ID must
  be passed to the code to generate a metatask for a specific OpenML task. Technically, any ID can be passed. In
  practice, only supervised classification tasks are supported so far. Moreover, this tool was build for and tested
  against curated benchmarks (like tasks in OpenMLCC-18). Other classification tasks should be supported as well but
  bugs might be more likely.
* **FakedClassifiers**: Code to simulate the behavior of a base model by passing appropriate data to it during the
  initialization. Allows us to evaluate most ensemble techniques without code changes to the original implementation's
  code.
* **Supporting Ensemble Techniques** We created code to make ensemble techniques usable with (pre-fitted) base models.
  This is not part of Assembled itself but rather additional example on how to use ensemble techniques with Assembled.
  Some implementation support base models by default other do not. See `/ensemble_techniques/` for more details.

Currently, is main use-cases are:

* Ensembles After AutoML (Post-Processing)

This repository/branch also contains the Assembled-OpenML extension.

## Publicly Available Data for Assembled

The followings projects collected data for assembled and share them publicly:

* Metatasks containing the data for base models produced by executing [AutoGluon](https://auto.gluon.ai/) on the 71
  classification datasets from the AutoML benchmark: [Code](https://doi.org/10.6084/m9.figshare.23609226)
  and [Data](https://figshare.com/articles/dataset/Metatasks_for_AutoGluon_-_ROC_AUC_and_Balanced_Accuracy/23609361)
* Metatasks containing the data for base models produced by
  executing [Auto-Sklearn 1](https://automl.github.io/auto-sklearn) on the 71 classification datasets from the AutoML
  benchmark: TBA

## Assembled-OpenML

_For the original code of the workshop paper on Assembled-OpenML, see the `automl_workshop_paper` branch_

Assembled-OpenML builds Metatasks from OpenML. In this first version of Assembled-OpenML, the predictions correspond to
the top-n best runs (configurations) of an OpenML task. It shall simulate the use case of post-processing an AutoML
tool's top-n set of configurations.

Assembled-OpenML enables the user to quickly generate a benchmark set by entering a list of OpenML Task IDs as input
(see our code examples). In general, Assembled-OpenML is an affordable/efficient alternative to creating benchmarks by
hand. It is affordable/efficient, because you do not need to train and evaluate the base models but can directly
evaluate ensemble techniques.

## Installation

To install Assembled and Assembled-OpenML, use:

```bash
pip install assembled[openml]
```

If you only want to use Assembled, leave away `[openml]`.

To install the newest version (from the main branch), use:

```bash
pip install git+https://github.com/ISG-Siegen/assembled.git#egg=assembled[openml]
```

### Other Installations

For experiments, work-in-progress code, or other non-packaged code stored in this repository, we
provide `requirements.txt` files. These can be used to re-create the environments needed for the code.

An example workflow for the installation on Linux is:

```bash
git clone https://github.com/ISG-Siegen/assembled.git
cd assembled
python3 -m venv venv_assembled
source venv_assembled/bin/activate
pip install -r requirements.txt
```

Please be aware that any relevant-enough subdirectory keeps track of its own requirements through a `requirements.txt`.
Hence, if you want to use only parts of this project, it might be a better idea to only install the requirements of the
code that you want to use.

## Usage

To see the example usage of Assembled-OpenML, see the `./examples/` directory for code examples and more details.

A simple example of using Assembled-OpenML to get a Metatask and using Assembled to evaluate an ensemble technique on
the Metatask is:

```python
from assembledopenml.openml_assembler import OpenMLAssembler
from assembled.ensemble_evaluation import evaluate_ensemble_on_metatask

# Import an adapted version of auto-sklearn's Ensemble Selection
# (requires the ensemble_techniques directory to be in your local directory)
from ensemble_techniques.autosklearn.ensemble_selection import EnsembleSelection
from ensemble_techniques.util.metrics import OpenMLAUROC

# -- Use Assembled-OpenML to build a metatask for the OpenML task with ID 3
omla = OpenMLAssembler(nr_base_models=50, openml_metric_name="area_under_roc_curve")
mt = omla.run(openml_task_id=3)

# -- Benchmark the ensemble technique on the metatask
technique_run_args = {"ensemble_size": 50, "metric": OpenMLAUROC}
fold_scores = evaluate_ensemble_on_metatask(mt, EnsembleSelection, technique_run_args, "autosklearn.EnsembleSelection",
                                            pre_fit_base_models=True,
                                            meta_train_test_split_fraction=0.5,
                                            meta_train_test_split_random_state=0,
                                            return_scores=OpenMLAUROC)
print(fold_scores)
print("Average Performance:", sum(fold_scores) / len(fold_scores))
```

## Limitations

* **Regression is not supported** so far as OpenML has not enough data (runs) on Regression tasks. Would require some
  additional implementations.
* Assembled-OpenML ignores OpenML repetitions (most runs/datasets do not provide repetitions).
* The file format for the predictions file is not fully standardized in OpenML and hence requires manually adjustment to
  all used formats. Hopefully, we found most of the relevant formats with Assembled-OpenML.
* Some files, which store predictions, seem to have malicious or corrupted predictions/confidence values. If we can not
  fix such a case, we store these bad predictors in the Metatask object to be manually validated later on. Moreover,
  these predictors can be filtered from the Metatask if one wants to (we do this for every example or experiment).

## A Comment on Validation Data

By default, and by design, Metatask created only from OpenML data do not have inner fold validation data. To train an
ensemble techniques on metataks created only from OpenML data, we split a fold's predictions on the fold's test data of
into ensemble_train and ensemble_test data. With ensemble_train being used to build/train the ensemble and ensemble_test
being used to evaluate the ensemble.

Alternatively, if a metatask and the base models stored in the metatask were initialized / created with validation data,
we can also use the validation data to train the ensemble technique and then test it on all test data/predictions of a
fold.

## Relevant Publication

If you use Assembled or Assembled-OpenML in scientific publications, we would appreciate citations.

**Assembled-OpenML: Creating Efficient Benchmarks for Ensembles in AutoML with OpenML**, _Lennart Purucker and Joeran
Beel,_
_First Conference on Automated Machine Learning (Late-Breaking Workshop), 2022_

Link to
publication: [AutoML Conference](https://2022.automl.cc/wp-content/uploads/2022/08/assembled_openml_creating_effi-Main-Paper-And-Supplementary-Material.pdf)
and arXiv (TBA)

Link to teaser video: [YouTube](https://www.youtube.com/watch?v=8OI8pWfWzM8)

Link to full video: [YouTube](https://www.youtube.com/watch?v=WC-ndeKr_Ms)

```
@inproceedings{purucker2022assembledopenml,
    title={Assembled-Open{ML}: Creating Efficient Benchmarks for Ensembles in Auto{ML} with Open{ML}},
    author={Lennart Purucker and Joeran Beel},
    booktitle={First Conference on Automated Machine Learning (Late-Breaking Workshop)},
    year={2022}
}
```
