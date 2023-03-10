[![PyPI](https://img.shields.io/pypi/v/dwave-scikit-learn-plugin.svg)](https://pypi.python.org/pypi/dwave-scikit-learn-plugin)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/dwavesystems/dwave-scikit-learn-plugin/tree/main.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/dwavesystems/dwave-scikit-learn-plugin)

# dwave-scikit-learn-plugin

This package contains a `sci-kit learn` transformer which does hybrid feature selection. For more information about the hybrid feature selection method, see the [example notebook](https://github.com/dwave-examples/feature-selection-notebook).

## Examples

The main class is a `sklearn.feature_selection.SelectorMixin` and so can be used in any existing sklearn pipeline.

A minimal example of using the hybrid feature selection: 

```python
from dwave.plugins.sklearn.transformers import SelectFromQuadraticModel
import numpy as np

# generate uniformly random data, 10,000 observations and 300 features 
data = np.random.uniform(-10,10, size = (10000,300))

outcome =  np.array([int(i) for i in (np.random.uniform(0,1, size = (10000,1)) > .5)])

# instantiate the feature selection class
selector = SelectFromQuadraticModel()

# do hybrid feature selection 
data_transformed = selector.fit_transform(data, outcome)
```

For large problems, the default runtime may be insufficient. You can use the CQM solvers's 
[`min_time_limit`](https://docs.ocean.dwavesys.com/en/stable/docs_system/reference/generated/dwave.system.samplers.LeapHybridCQMSampler.min_time_limit.html)
method to find the minimum accepted runtime for your problem; alternatively, simply submit as above 
and check the returned error message for the required runtime. 

The feature selector can be re-instantiated with a longer time limit.

```python
# instantiate the feature selection class with a longer time limit 
selector = SelectFromQuadraticModel(time_limit = 200) 
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
