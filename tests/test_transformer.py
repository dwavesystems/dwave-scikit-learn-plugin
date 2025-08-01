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

import concurrent.futures
import unittest
import unittest.mock
import warnings
import tempfile
from parameterized import parameterized

import dimod
import numpy as np

from dwave.optimization import Model
from dwave.cloud.exceptions import ConfigFileError, SolverAuthenticationError
from dwave.system import LeapHybridNLSampler
from dwave.system import LeapHybridCQMSampler
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from dwave.plugins.sklearn.transformers import SelectFromQuadraticModel
from dwave.plugins.sklearn.utilities import corrcoef

FEASIBLE_NL_SOLUTION = []

class MockCQM(dimod.ExactCQMSolver):
    def sample_cqm(self, cqm: dimod.CQM, *, time_limit: float, label: str) -> dimod.SampleSet:
        return super().sample_cqm(cqm)

    def min_time_limit(self, cqm):
        return 1


class MockNL():
    def sample(self, nl: Model, *, time_limit: float, label: str):
        nl.states.resize(1)

        for decision in nl.iter_decisions():
            decision.set_state(0, FEASIBLE_NL_SOLUTION)

        return concurrent.futures.Future()


@unittest.mock.patch("dwave.plugins.sklearn.transformers.LeapHybridNLSampler", MockNL)
@unittest.mock.patch("dwave.plugins.sklearn.transformers.LeapHybridCQMSampler", MockCQM)
class TestSelectFromQuadraticModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(138984)
        cls.X = rng.uniform(-10, 10, size=(100, 9))
        cls.y = np.asarray(rng.uniform(0, 1, size=100) > 0.5, dtype=int)
    
    @parameterized.expand([
        (0.1, 30, "cqm"), 
        (0.1, 15, "cqm"), 
        (0.1, 30, "nl"), 
        (0.1, 15, "nl"), 
    ])
    def test_init_good(self, alpha, time_limit, solver):
        a = SelectFromQuadraticModel(solver=solver)

        b = SelectFromQuadraticModel(alpha=alpha, solver=solver)

        c = SelectFromQuadraticModel(alpha=alpha, time_limit=time_limit, solver=solver)

        d = SelectFromQuadraticModel(time_limit=time_limit, solver=solver)

        self.assertIsInstance(a, SelectFromQuadraticModel)
        self.assertIsInstance(b, SelectFromQuadraticModel)
        self.assertIsInstance(c, SelectFromQuadraticModel)
        self.assertIsInstance(d, SelectFromQuadraticModel)

        self.assertEqual(a.alpha, 0.5)
        self.assertEqual(b.alpha, 0.1)
        self.assertEqual(c.alpha, 0.1)
        self.assertEqual(d.alpha, 0.5)

        self.assertEqual(a.time_limit, None)
        self.assertEqual(b.time_limit, None)
        self.assertEqual(c.time_limit, time_limit)
        self.assertEqual(d.time_limit, time_limit)

        self.assertIsInstance(
            SelectFromQuadraticModel(alpha=0), SelectFromQuadraticModel
        )

    @parameterized.expand([
        (-10, "cqm"), 
        (10, "cqm"), 
        (-10, "nl"), 
        (10, "nl"), 
    ])
    def test_init_bad(self, alpha, solver):
        self.assertRaises(ValueError, SelectFromQuadraticModel, alpha=alpha, solver=solver)
        self.assertRaises(ValueError, SelectFromQuadraticModel, alpha=alpha, solver=solver)
    
    @parameterized.expand([
        (7, "cqm"), 
        (5, "cqm"), 
        (7, "nl"), 
        (5, "nl"), 
    ])
    def test_fit(self, num_features, solver):
        global FEASIBLE_NL_SOLUTION
        if num_features==7:
            FEASIBLE_NL_SOLUTION = [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0]
        elif num_features==5:
            FEASIBLE_NL_SOLUTION = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]

        selector = SelectFromQuadraticModel(num_features=num_features, solver=solver)

        # test default numpy

        selector.fit(self.X, self.y)
        self.assertEqual(sum(selector._mask), num_features)

        try:
            self.X[:, selector._mask]
        except Exception as e:
            self.fail(e)

        # test non-default numpy

        selector.fit(self.X, self.y, num_features=num_features, solver=solver)
        self.assertEqual(sum(selector._mask), num_features)

        try:
            self.X[:, selector._mask]
        except Exception as e:
            self.fail(e)
    
    @parameterized.expand([
        (7, "cqm"), 
        (7, "nl"), 
    ])
    def test_fit_transform(self, num_features, solver):
        global FEASIBLE_NL_SOLUTION 
        FEASIBLE_NL_SOLUTION = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]

        selector = SelectFromQuadraticModel(num_features=num_features, solver=solver)

        # test numpy without fit
        x = selector.fit_transform(self.X, self.y, num_features=num_features-2, solver=solver)

        self.assertEqual(x.shape[1], num_features-2)

        x_from_fit = self.X[:, selector._mask]
        np.testing.assert_array_equal(x, x_from_fit)
    
    @parameterized.expand([
        (2, "cqm"), 
        (2, "nl"), 
    ])
    def test_pipeline(self, num_features, solver):
        global FEASIBLE_NL_SOLUTION 
        FEASIBLE_NL_SOLUTION = [1.0, 1.0, 0.0, 0.0]

        X, y = load_iris(return_X_y=True)

        clf = Pipeline([
          ('feature_selection', SelectFromQuadraticModel(num_features=num_features, solver=solver)),
          ('classification', RandomForestClassifier())
        ])
        clf.fit(X, y)

        clf.predict(X)
    
    def test_alpha_0(self):
        cqm = SelectFromQuadraticModel.correlation(self.X, self.y, num_features=3, alpha=0, solver="cqm")
        self.assertTrue(not any(cqm.objective.linear.values()))

        X = np.atleast_2d(np.asarray(self.X))
        y = np.asarray(self.y)

        with tempfile.TemporaryFile() as fX, tempfile.TemporaryFile() as fout:
            # we make a copy of X because we'll be modifying it in-place within
            # some of the functions
            X_copy = np.memmap(fX, X.dtype, mode="w+", shape=(X.shape[0], X.shape[1] + 1))
            X_copy[:, :-1] = X
            X_copy[:, -1] = y

            # make the matrix that will hold the correlations
            correlations = np.memmap(
                fout,
                dtype=np.result_type(X, y),
                mode="w+",
                shape=(X_copy.shape[1], X_copy.shape[1]),
                )

            # main calculation. It modifies X_copy in-place
            corrcoef(X_copy, out=correlations, rowvar=False, copy=False)

            # we don't care about the direction of correlation in terms of
            # the penalty/quality
            np.absolute(correlations, out=correlations)

        label_corr = np.array(correlations[:-1,-1])
        expected_linear = np.zeros(X.shape[1])
        expected_linear += (-1.0 * label_corr * 0 * 3)
        self.assertTrue(np.allclose(expected_linear.all(), 0))
    
    @parameterized.expand([
        (3, 1, "cqm"), 
        (3, 1, "nl"), 
    ])
    def test_alpha_1(self, num_features, alpha, solver):
        global FEASIBLE_NL_SOLUTION
        FEASIBLE_NL_SOLUTION = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        rng = np.random.default_rng(42)

        y = rng.uniform(size=1000)

        # make the first three columns exactly match the test data
        X = rng.uniform(size=(1000, 10))
        X[:, 0] = X[:, 1] = X[:, 2] = y

        selector = SelectFromQuadraticModel(num_features=num_features, alpha=alpha, solver=solver).fit(X, y)

        # with alpha=1, we should see that only the quality matters, so the
        # first three should be selected despite being perfectly correlated
        self.assertTrue(selector._get_support_mask()[0:3].all())
        self.assertFalse(selector._get_support_mask()[3:].any())

    @parameterized.expand([
        (1, "cqm"), 
        (1, "nl"), 
    ])
    def test_xy_shape(self, num_features, solver):
        with self.assertRaises(ValueError):
            SelectFromQuadraticModel(num_features=num_features, solver=solver).fit([[0, 1]], [1, 2])

    def test_repr(self):
        repr(SelectFromQuadraticModel(solver="cqm"))
        repr(SelectFromQuadraticModel(solver="nl"))

    @parameterized.expand([
        (2, "cqm"), 
        (2, "nl"), 
    ])
    def test_gridsearch(self, num_features, solver):
        global FEASIBLE_NL_SOLUTION
        FEASIBLE_NL_SOLUTION = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
        rng = np.random.default_rng(42)
        X = rng.uniform(-10, 10, size=(100, 9))
        y = np.asarray(rng.uniform(0, 1, size=100) > 0.5, dtype=int)

        pipe = Pipeline([
          ('feature_selection', SelectFromQuadraticModel(num_features=num_features, solver=solver)),
          ('classification', RandomForestClassifier())
        ])

        clf = GridSearchCV(pipe,
                           param_grid=dict(
                            feature_selection__num_features=[num_features+1],
                            feature_selection__alpha=[0, .5]))
        clf.fit(X, y)

    @parameterized.expand([
        (2, "cqm"), 
        (2, "nl"), 
    ])
    def test_one_row(self, num_features, solver):
        X = [[-7.85717866, 1.93442648, 8.85760003]]
        y = [1]

        with self.assertRaises(ValueError):
            SelectFromQuadraticModel(num_features=num_features, solver=solver).fit(X, y)

    def test_fixed_column(self):
        global FEASIBLE_NL_SOLUTION
        FEASIBLE_NL_SOLUTION = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        X = np.copy(self.X)

        # fix two of the columns
        X[:, 1] = 0
        X[:, 5] = 1

        cqm = SelectFromQuadraticModel.correlation(X, self.y, alpha=0.5, num_features=5, solver="cqm")
        fitted = SelectFromQuadraticModel(alpha=0.5, num_features=5, solver="cqm").fit(X, self.y)

        # in this case the linear bias for those two columns should be 0
        self.assertEqual(cqm.objective.linear[1], 0)
        self.assertEqual(cqm.objective.linear[5], 0)

        # as should the quadratic biases
        self.assertEqual(cqm.objective.degree(1), 0)
        self.assertEqual(cqm.objective.degree(5), 0)

        selected = SelectFromQuadraticModel(alpha=0.5, num_features=5, solver="nl").fit(X, self.y)

        # Check that the variables corresponding to constant columns are not present
        self.assertEqual(selected._mask.all(), fitted._mask.all())

class TestIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            LeapHybridNLSampler()
            LeapHybridCQMSampler()
        except (ConfigFileError, SolverAuthenticationError, ValueError):
            raise unittest.SkipTest("no hybrid solver available")

    def test_pipeline(self):
        X, y = load_iris(return_X_y=True)

        clf = Pipeline([
          ('feature_selection', SelectFromQuadraticModel(num_features=2)),
          ('classification', RandomForestClassifier())
        ])
        clf.fit(X, y)

        clf.predict(X)
