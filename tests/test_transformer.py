# Copyright 523 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import unittest
import unittest.mock
import warnings

import numpy as np

from dwave.optimization import Model
from dwave.cloud.exceptions import ConfigFileError, SolverAuthenticationError
from dwave.system import LeapHybridNLSampler
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from dwave.plugins.sklearn.transformers import SelectFromNonlinearModel


class MockNL():
    def sample(self, NL: Model, *, time_limit: float, label: str):
        sampler = LeapHybridNLSampler()
        return sampler.sample(NL)

    def min_time_limit(self, NL):
        return 1


@unittest.mock.patch("dwave.plugins.sklearn.transformers.LeapHybridNLSampler", MockNL)
class TestSelectFromNonlinearModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(138984)
        cls.X = rng.uniform(-10, 10, size=(100, 9))
        cls.y = np.asarray(rng.uniform(0, 1, size=100) > 0.5, dtype=int)

    def test_init_good(self):
        a = SelectFromNonlinearModel()

        b = SelectFromNonlinearModel(alpha=0.1)

        c = SelectFromNonlinearModel(alpha=0.1, time_limit=30)

        d = SelectFromNonlinearModel(time_limit=15)

        self.assertIsInstance(a, SelectFromNonlinearModel)
        self.assertIsInstance(b, SelectFromNonlinearModel)
        self.assertIsInstance(c, SelectFromNonlinearModel)
        self.assertIsInstance(d, SelectFromNonlinearModel)

        self.assertEqual(a.alpha, 0.5)
        self.assertEqual(b.alpha, 0.1)
        self.assertEqual(c.alpha, 0.1)
        self.assertEqual(d.alpha, 0.5)

        self.assertEqual(a.time_limit, None)
        self.assertEqual(b.time_limit, None)
        self.assertEqual(c.time_limit, 30)
        self.assertEqual(d.time_limit, 15)

        self.assertIsInstance(
            SelectFromNonlinearModel(alpha=0), SelectFromNonlinearModel
        )

    def test_init_bad(self):
        self.assertRaises(ValueError, SelectFromNonlinearModel, alpha=-10)
        self.assertRaises(ValueError, SelectFromNonlinearModel, alpha=10)

    def test_fit(self):
        selector = SelectFromNonlinearModel(num_features=7)

        # test default numpy

        selector.fit(self.X, self.y)
        self.assertEqual(sum(selector._mask), 7)

        try:
            self.X[:, selector._mask]
        except Exception as e:
            self.fail(e)

        # test non-default numpy

        selector.fit(self.X, self.y, num_features=5)
        self.assertEqual(sum(selector._mask), 5)

        try:
            self.X[:, selector._mask]
        except Exception as e:
            self.fail(e)
    
    def test_fit_transform(self):
        selector = SelectFromNonlinearModel(num_features=7)

        # test numpy without fit
        x = selector.fit_transform(self.X, self.y, num_features=5)

        self.assertEqual(x.shape[1], 5)

        x_from_fit = self.X[:, selector._mask]
        np.testing.assert_array_equal(x, x_from_fit)

    def test_pipeline(self):
        X, y = load_iris(return_X_y=True)

        clf = Pipeline([
          ('feature_selection', SelectFromNonlinearModel(num_features=2)),
          ('classification', RandomForestClassifier())
        ])
        clf.fit(X, y)

        clf.predict(X)
    
    def test_alpha_0(self):
        NL, _, linear = SelectFromNonlinearModel.correlation_nl(self.X, self.y, num_features=3, alpha=0)
        self.assertTrue(np.allclose(linear, 0))
    
    def test_alpha_1(self):
        rng = np.random.default_rng(42)

        y = rng.uniform(size=1000)

        # make the first three columns exactly match the test data
        X = rng.uniform(size=(1000, 10))
        X[:, 0] = X[:, 1] = X[:, 2] = y

        selector = SelectFromNonlinearModel(num_features=3, alpha=1).fit(X, y)

        # with alpha=1, we should see that only the quality matters, so the
        # first three should be selected despite being perfectly correlated
        self.assertTrue(selector._get_support_mask()[0:3].all())
        self.assertFalse(selector._get_support_mask()[3:].any())

    def test_xy_shape(self):
        with self.assertRaises(ValueError):
            SelectFromNonlinearModel(num_features=1).fit([[0, 1]], [1, 2])

    def test_repr(self):
        repr(SelectFromNonlinearModel())

    def test_gridsearch(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(-10, 10, size=(100, 9))
        y = np.asarray(rng.uniform(0, 1, size=100) > 0.5, dtype=int)

        pipe = Pipeline([
          ('feature_selection', SelectFromNonlinearModel(num_features=2)),
          ('classification', RandomForestClassifier())
        ])

        clf = GridSearchCV(pipe,
                           param_grid=dict(
                            feature_selection__num_features=[3],
                            feature_selection__alpha=[0, .5]))
        clf.fit(X, y)

    def test_one_row(self):
        X = [[-7.85717866, 1.93442648, 8.85760003]]
        y = [1]

        with self.assertRaises(ValueError):
            SelectFromNonlinearModel(num_features=2).fit(X, y)
    
    def test_fixed_column(self):
        X = np.copy(self.X)

        # fix two of the columns
        X[:, 1] = 0
        X[:, 5] = 1

        NL, X_binary = SelectFromNonlinearModel.correlation_nl(X, self.y, alpha=.5, num_features=5)

        # Convert the objective expression to a string for symbol inspection
        objective_str = str(NL.objective)

        # Check that the variables corresponding to constant columns are not present
        for col in [1, 5]:
            var_name = str(X_binary[col])
            self.assertNotIn(var_name, objective_str, msg=f"Constant column {col} still appears in objective.")
    

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            LeapHybridNLSampler()
        except (ConfigFileError, SolverAuthenticationError, ValueError):
            raise unittest.SkipTest("no hybrid solver available")

    def test_pipeline(self):
        X, y = load_iris(return_X_y=True)

        clf = Pipeline([
          ('feature_selection', SelectFromNonlinearModel(num_features=2)),
          ('classification', RandomForestClassifier())
        ])
        clf.fit(X, y)

        clf.predict(X)
