[![PyPI](https://img.shields.io/pypi/v/dwave-scikit-learn-plugin.svg)](https://pypi.python.org/pypi/dwave-scikit-learn-plugin)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/dwavesystems/dwave-scikit-learn-plugin/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/dwavesystems/dwave-scikit-learn-plugin)

# D-Wave `scikit-learn` Plugin

This package provides a [scikit-learn](https://scikit-learn.org/) transformer for 
[feature selection](https://en.wikipedia.org/wiki/Feature_selection) using a
quantum-classical [hybrid solver](https://docs.ocean.dwavesys.com/en/stable/concepts/hybrid.html).

This plugin makes use of a Leap™ quantum-classical hybrid solver. Developers can get started by
[signing up](https://cloud.dwavesys.com/leap/signup/) for the Leap quantum cloud service for free.
Those seeking a more collaborative approach and assistance with building a production application can
reach out to D-Wave [directly](https://www.dwavesys.com/solutions-and-products/professional-services/) and also explore the feature selection [offering](https://aws.amazon.com/marketplace/pp/prodview-bsrc3yuwgjbo4) in AWS Marketplace.

The package's main class, `SelectFromNonlinearModel`, can be used in any existing `sklearn` pipeline.
For an introduction to hybrid methods for feature selection, see the [Feature Selection for Nonlinear Models](https://github.com/dwave-examples/feature-selection-cqm).

## Examples

### Basic Usage

A minimal example of using the plugin to select 20 of 30 features of an `sklearn` dataset: 

```python
>>> from sklearn.datasets import load_breast_cancer
>>> from dwave.plugins.sklearn import SelectFromNonlinearModel
... 
>>> X, y = load_breast_cancer(return_X_y=True)
>>> X.shape
(569, 30)
>>> X_new = SelectFromNonlinearModel(num_features=20).fit_transform(X, y)
>>> X_new.shape
(569, 20)
```

For large problems, the default runtime may be insufficient. You can use the nonlinear (NL) solver's 
[`time_limit`](https://docs.ocean.dwavesys.com/en/stable/docs_system/reference/generated/dwave.system.samplers.LeapHybridCQMSampler.min_time_limit.html)
method to find the minimum accepted runtime for your problem; alternatively, simply submit as above 
and check the returned error message for the required runtime. 

The feature selector can be re-instantiated with a longer time limit.

```python
>>> X_new = SelectFromNonlinearModel(num_features=20, time_limit=200).fit_transform(X, y)
```

### Tuning

You can use `SelectFromNonlinearModel` with scikit-learn's
[hyper-parameter optimizers](https://scikit-learn.org/stable/modules/classes.html#hyper-parameter-optimizers).

For example, the number of features can be tuned using a grid search. **Please note that this will
submit many problems to the hybrid solver.**

```python
>>> import numpy as np
...
>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.model_selection import GridSearchCV
>>> from sklearn.pipeline import Pipeline
>>> from dwave.plugins.sklearn import SelectFromNonlinearModel
...
>>> X, y = load_breast_cancer(return_X_y=True)
...
>>> num_features = X.shape[1]
>>> searchspace = np.linspace(1, num_features, num=5, dtype=int, endpoint=True)
...
>>> pipe = Pipeline([
>>>   ('feature_selection', SelectFromNonlinearModel()),
>>>   ('classification', RandomForestClassifier())
>>> ])
...
>>> clf = GridSearchCV(pipe, param_grid=dict(feature_selection__num_features=searchspace))
>>> search = clf.fit(X, y)
>>> print(search.best_params_)
{'feature_selection__num_features': 22}
```

## Installation

To install the core package:

```bash
pip install dwave-scikit-learn-plugin
```

## License

Released under the Apache License 2.0

## Contributing

Ocean's [contributing guide](https://docs.ocean.dwavesys.com/en/stable/contributing.html)
has guidelines for contributing to Ocean packages.

### Release Notes

**dwave-scikit-learn-plugin** makes use of [reno](https://docs.openstack.org/reno/) to manage its
release notes.

When making a contribution to **dwave-scikit-learn-plugin** that will affect users, create a new
release note file by running

```bash
reno new your-short-descriptor-here
```

You can then edit the file created under ``releasenotes/notes/``.
Remove any sections not relevant to your changes.
Commit the file along with your changes.
