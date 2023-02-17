# For Developers

This section is very much work in progress...

## Unit Tests

We use [py.test](https://docs.pytest.org/en/latest/) for unit testing.

While in the root directory (and in the current environment), call:

```bash
python -m pytest tests/
```

## Pre commits

Install the pre-commit hook while being in the environment with:

```bash
pre-commit install
```

Afterwards, each time you commit, the pre-commit hook will be executed.

Use `pre-commit run --all-files` to run it for all files.

Which hooks to use is still in debate. Not sure about mypy default etc.

## Some Notes on Design Decision

* We do not support repetitions in a metatask itself. We argue that in such case the repetitions are "outside" of the
  metatasks. In other words, to benchmark n-repeated k-fold we think it is more appropriate to create n metatasks
  instead of 1 containing all repetition data. Not sure if this is the best way for the future but for now it is okay.
    * Including n-repeated in a metatask could be achieved by adding appropriate prefixes to the base models and making
      the fold_indicator a 2D array.

## Build and Deployment

* While in the project root call: `python3 -m build`
* Afterwards, upload it to pypi via: `python3 -m twine upload dist/*`

## Other

### Dataset Checks for Fake Base Models
Since many ensemble techniques would not need to touch the trainings features, we could
avoid preprocessing the training features entirely for these cases. This, however, requires
us to remove a lot of checks from our models and code (like no nans allowed, numeric only dtype,...).

Since we do not want to do this, it seems more logical to determine a default preprocessor.
The default preprocessor transforms categories to integers and fills missing values.

# Documentation TODOs for the Future

* Setup Docstring Documentation / Webpage
* Make User and Developer Documentation Separate; add more details and examples -> extensive user / developer
  documentation
* Refactor / Re-work unit test to exclude OpenML as much as possible and add more tests
* Add CI: Automatic Testing; Releases
