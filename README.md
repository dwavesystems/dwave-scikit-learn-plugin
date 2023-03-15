[![PyPI](https://img.shields.io/pypi/v/dwave-scikit-learn-plugin.svg)](https://pypi.python.org/pypi/dwave-scikit-learn-plugin)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/dwavesystems/dwave-scikit-learn-plugin/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/dwavesystems/dwave-scikit-learn-plugin)

# D-Wave `scikit-learn` Plugin

This package provides a [scikit-learn](https://scikit-learn.org/) transformer for 
[feature selection](https://en.wikipedia.org/wiki/Feature_selection) using a quantum-classical hybrid solver. 

For an introduction to hybrid methods for feature selection, see the [Feature Selection example](https://github.com/dwave-examples/feature-selection-notebook) Jupyter Notebook.

The package's main class, `SelectFromQuadraticModel`, can be used in any existing `sklearn` pipeline.

## Examples

A minimal example of using the plugin: 

```python
from dwave.plugins.sklearn.transformers import SelectFromQuadraticModel
import numpy as np

# generate uniformly random data, 10,000 observations and 300 features
data = np.random.uniform(-10, 10, size=(10_000, 300))

outcome = np.asarray(np.random.uniform(0, 1, size=10_000) > 0.5, dtype=int)

# instantiate the feature selection class
selector = SelectFromQuadraticModel()

# solve the feature-selection problem
data_transformed = selector.fit_transform(data, outcome)
```

For large problems, the default runtime may be insufficient. You can use the CQM solver's 
[`min_time_limit`](https://docs.ocean.dwavesys.com/en/stable/docs_system/reference/generated/dwave.system.samplers.LeapHybridCQMSampler.min_time_limit.html)
method to find the minimum accepted runtime for your problem; alternatively, simply submit as above 
and check the returned error message for the required runtime. 

The feature selector can be re-instantiated with a longer time limit.

```python
# instantiate the feature selection class with a longer time limit
selector = SelectFromQuadraticModel(time_limit=200)
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
