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
from dwave.plugins.sklearn.transformers import SelectFromQuadraticModel

import numpy as np
import pandas as pd
import dimod


class MockCQM(dimod.ExactCQMSolver):
    def __init__(self):
        super().__init__()

    def min_time_limit(self, cqm):
        return 1


@unittest.mock.patch("dwave.plugins.sklearn.transformers.LeapHybridCQMSampler", MockCQM)
class TestTransformer(unittest.TestCase):
    def __init__(self, methodName: str = None) -> None:
        super().__init__(methodName)
        self.rng = np.random.default_rng(138984)
        self.X_np = None
        self.y_np = None
        self.X_pd = None
        self.y_pd = None

    def create_data_numpy(self) -> None:
        """Idempotent function that instantiates a class variable containing test numpy data"""

        if self.X_np is None:
            self.X_np = self.rng.uniform(-10, 10, size=(100, 9))

        if self.y_np is None:
            self.y_np = np.array(
                [int(i) for i in (self.rng.uniform(0, 1, size=(100, 1)) > 0.5)]
            )

    def create_data_pd(self) -> None:
        """Idempotent function that instantiates a class variable containing test pandas data
        derived from the numpy data. If `create_data_numpy` has not been called, this function
        will call it.
        """
        self.create_data_numpy()

        if self.X_pd is None:
            self.X_pd = pd.DataFrame(self.X_np)

        if self.y_pd is None:
            self.y_pd = pd.DataFrame(self.y_np)

    def setUp(self) -> None:
        super().setUp()
        self.create_data_pd()

    def test_init_good(self):
        """Enforcing defaults and making sure variables are set"""
        a = SelectFromQuadraticModel()

        b = SelectFromQuadraticModel(alpha=0.1)

        c = SelectFromQuadraticModel(alpha=0.1, time_limit=30)

        d = SelectFromQuadraticModel(time_limit=15)

        self.assertIsInstance(a, SelectFromQuadraticModel)
        self.assertIsInstance(b, SelectFromQuadraticModel)
        self.assertIsInstance(c, SelectFromQuadraticModel)
        self.assertIsInstance(d, SelectFromQuadraticModel)

        self.assertEqual(a.alpha, 0.5)
        self.assertEqual(b.alpha, 0.1)
        self.assertEqual(c.alpha, 0.1)
        self.assertEqual(d.alpha, 0.5)

        self.assertEqual(a.time_limit, 10)
        self.assertEqual(b.time_limit, 10)
        self.assertEqual(c.time_limit, 30)
        self.assertEqual(d.time_limit, 15)

        self.assertIsInstance(
            SelectFromQuadraticModel(alpha=0), SelectFromQuadraticModel
        )

    def test_init_bad(self):
        """Testing edges of initialization parameters"""

        self.assertRaises(ValueError, SelectFromQuadraticModel, alpha=-10)
        self.assertRaises(ValueError, SelectFromQuadraticModel, alpha=10)
        self.assertRaises(ValueError, SelectFromQuadraticModel, time_limit=-100)
        self.assertRaises(ValueError, SelectFromQuadraticModel, time_limit=0)
        self.assertRaises(ValueError, SelectFromQuadraticModel, time_limit=1)

    def test_fit_no_y(self):
        """Test the fit method without an outcome variable specified"""

        selector = SelectFromQuadraticModel(n_default_feature=7)

        # test default numpy

        selector.fit(self.X_np)
        self.assertEqual(len(selector.selected_columns), 7)

        try:
            self.X_np[:, selector.selected_columns]
        except Exception as e:
            self.fail(e)

        # test default pandas
        selector.fit(self.X_pd)
        self.assertEqual(len(selector.selected_columns), 7)

        try:
            self.X_pd.loc[:, selector.selected_columns]
        except Exception as e:
            self.fail(e)

        # test non-default numpy

        selector.fit(self.X_np, number_of_features=5)
        self.assertEqual(len(selector.selected_columns), 5)

        try:
            self.X_np[:, selector.selected_columns]
        except Exception as e:
            self.fail(e)

        # test non-default pandas
        selector.fit(self.X_pd, number_of_features=5)
        self.assertEqual(len(selector.selected_columns), 5)

        try:
            self.X_pd.loc[:, selector.selected_columns]
        except Exception as e:
            self.fail(e)

        # test non-strict fit numpy
        selector.fit(self.X_np, strict=False)
        self.assertLessEqual(len(selector.selected_columns), 7)

        try:
            self.X_np[:, selector.selected_columns]
        except Exception as e:
            self.fail(e)

        # test non-strict fit pandas
        selector.fit(self.X_pd, strict=False)
        self.assertLessEqual(len(selector.selected_columns), 7)

        try:
            self.X_pd.loc[:, selector.selected_columns]
        except Exception as e:
            self.fail(e)

    def test_fit_y(self):
        """Test the fit method with an outcome variable specified."""
        selector = SelectFromQuadraticModel(n_default_feature=7)

        # test default numpy

        selector.fit(self.X_np, self.y_np)
        self.assertEqual(len(selector.selected_columns), 7)

        try:
            self.X_np[:, selector.selected_columns]
        except Exception as e:
            self.fail(e)

        # test default pandas
        selector.fit(self.X_pd, self.y_pd)
        self.assertEqual(len(selector.selected_columns), 7)

        try:
            self.X_pd.loc[:, selector.selected_columns]
        except Exception as e:
            self.fail(e)

        # test non-default numpy

        selector.fit(self.X_np, self.y_np, number_of_features=5)
        self.assertEqual(len(selector.selected_columns), 5)

        try:
            self.X_np[:, selector.selected_columns]
        except Exception as e:
            self.fail(e)

        # test non-default pandas
        selector.fit(self.X_pd, self.y_pd, number_of_features=5)
        self.assertEqual(len(selector.selected_columns), 5)

        try:
            self.X_pd.loc[:, selector.selected_columns]
        except Exception as e:
            self.fail(e)

        # test non-strict fit numpy
        selector.fit(self.X_np, self.y_np, strict=False)
        self.assertLessEqual(len(selector.selected_columns), 7)

        try:
            self.X_np[:, selector.selected_columns]
        except Exception as e:
            self.fail(e)

        # test non-strict fit pandas
        selector.fit(self.X_pd, self.y_pd, strict=False)
        self.assertLessEqual(len(selector.selected_columns), 7)

        try:
            self.X_pd.loc[:, selector.selected_columns]
        except Exception as e:
            self.fail(e)

    def test_transform_no_y(self):
        """Test the transform method without the outcome variable specified"""
        selector = SelectFromQuadraticModel(n_default_feature=7)

        # test numpy
        selector.fit(self.X_np, number_of_features=5)

        x = selector.transform(self.X_np)

        self.assertEqual(x.shape[1], 5)

        x_from_fit = self.X_np[:, selector.selected_columns]
        np.testing.assert_array_equal(x, x_from_fit)

        # test pandas
        selector.fit(self.X_pd, number_of_features=5)

        x = selector.transform(self.X_pd)

        self.assertEqual(x.shape[1], 5)

        x_from_fit = self.X_pd.loc[:, selector.selected_columns]
        self.assertTrue(x.equals(x_from_fit))

    def test_transform_y(self):
        """Test the transform method with the outcome variable specified"""
        selector = SelectFromQuadraticModel(n_default_feature=7)

        # test numpy
        selector.fit(self.X_np, self.y_np, number_of_features=5)

        x = selector.transform(self.X_np, self.y_np)

        self.assertEqual(x.shape[1], 5)

        x_from_fit = self.X_np[:, selector.selected_columns]
        np.testing.assert_array_equal(x, x_from_fit)

        # test pandas
        selector.fit(self.X_pd, self.y_pd, number_of_features=5)

        x = selector.transform(self.X_pd, self.y_pd)

        self.assertEqual(x.shape[1], 5)

        x_from_fit = self.X_pd.loc[:, selector.selected_columns]
        self.assertTrue(x.equals(x_from_fit))

        # test numpy without fit
        selector.unfit()

        x = selector.transform(self.X_np, self.y_np, number_of_features=5)

        self.assertEqual(x.shape[1], 5)

        x_from_fit = self.X_np[:, selector.selected_columns]
        np.testing.assert_array_equal(x, x_from_fit)

        # test pandas withoput fit
        selector.unfit()

        x = selector.transform(self.X_pd, self.y_pd, number_of_features=5)

        self.assertEqual(x.shape[1], 5)

        x_from_fit = self.X_pd.loc[:, selector.selected_columns]
        self.assertTrue(x.equals(x_from_fit))

    def test_fit_transform_no_y(self):
        """Test the transform method without the outcome variable specified"""
        selector = SelectFromQuadraticModel(n_default_feature=7)

        # test numpy without fit
        x = selector.fit_transform(self.X_np, number_of_features=5)

        self.assertEqual(x.shape[1], 5)

        x_from_fit = self.X_np[:, selector.selected_columns]

        np.testing.assert_array_equal(x, x_from_fit)

        # test pandas withoput fit
        x = selector.fit_transform(self.X_pd, number_of_features=5)

        self.assertEqual(x.shape[1], 5)

        x_from_fit = self.X_pd.loc[:, selector.selected_columns]
        self.assertTrue(x.equals(x_from_fit))

    def test_fit_transform_y(self):
        """Test the transform method with the outcome variable specified"""
        selector = SelectFromQuadraticModel(n_default_feature=7)

        # test numpy without fit
        x = selector.fit_transform(self.X_np, self.y_np, number_of_features=5)

        self.assertEqual(x.shape[1], 5)

        x_from_fit = self.X_np[:, selector.selected_columns]
        np.testing.assert_array_equal(x, x_from_fit)

        # test pandas withoput fit
        x = selector.fit_transform(self.X_pd, self.y_pd, number_of_features=5)

        self.assertEqual(x.shape[1], 5)

        x_from_fit = self.X_pd.loc[:, selector.selected_columns]
        self.assertTrue(x.equals(x_from_fit))

    def test_unfit(self):
        """Test `unfit` function"""
        selector = SelectFromQuadraticModel(n_default_feature=7)

        selector.fit(self.X_np, self.y_np)

        selector.unfit()

        self.assertIsNone(selector.selected_columns)

    def test_update_time_limit(self):
        """Test the `update_time_limit` function."""
        selector = SelectFromQuadraticModel(n_default_feature=7)
        selector.update_time_limit(100)
        self.assertEqual(selector.time_limit, 100)


@unittest.mock.patch("dwave.plugins.sklearn.transformers.LeapHybridCQMSampler", MockCQM)
class TestManyFeatures(unittest.TestCase):
    def __init__(self, methodName: str = None) -> None:
        super().__init__(methodName)
        self.rng = np.random.default_rng(1023884)
        self.X_np = None
        self.y_np = None
        self.X_pd = None
        self.y_pd = None

    def create_data_numpy(self) -> None:
        """Idempotent function that instantiates a class variable containing test numpy data"""

        if self.X_np is None:
            self.X_np = self.rng.uniform(-10, 10, size=(100, 9))

        if self.y_np is None:
            self.y_np = np.array(
                [int(i) for i in (self.rng.uniform(0, 1, size=(100, 1)) > 0.5)]
            )

        return None

    def create_data_pd(self) -> None:
        """Idempotent function that instantiates a class variable containing test pandas data
        derived from the numpy data. If `create_data_numpy` has not been called, this function
        will call it.
        """
        self.create_data_numpy()

        if self.X_pd is None:
            self.X_pd = pd.DataFrame(self.X_np)

        if self.y_pd is None:
            self.y_pd = pd.DataFrame(self.y_np)

    def setUp(self) -> None:
        super().setUp()
        self.create_data_pd()

    def test_fit_transform_no_y(self):
        """Test the transform method without the outcome variable specified"""
        selector = SelectFromQuadraticModel(chunksize=2, n_default_feature=7)

        # test numpy without fit
        x = selector.fit_transform(self.X_np, number_of_features=5)

        self.assertEqual(x.shape[1], 5)

        x_from_fit = self.X_np[:, selector.selected_columns]

        np.testing.assert_array_equal(x, x_from_fit)

        # test pandas withoput fit
        x = selector.fit_transform(self.X_pd, number_of_features=5)

        self.assertEqual(x.shape[1], 5)

        x_from_fit = self.X_pd.loc[:, selector.selected_columns]
        self.assertTrue(x.equals(x_from_fit))

    def test_fit_transform_y(self):
        """Test the transform method with the outcome variable specified"""
        selector = SelectFromQuadraticModel(chunksize=2, n_default_feature=7)

        # test numpy without fit
        x = selector.fit_transform(self.X_np, self.y_np, number_of_features=5)

        self.assertEqual(x.shape[1], 5)

        x_from_fit = self.X_np[:, selector.selected_columns]
        np.testing.assert_array_equal(x, x_from_fit)

        # test pandas withoput fit
        x = selector.fit_transform(self.X_pd, self.y_pd, number_of_features=5)

        self.assertEqual(x.shape[1], 5)

        x_from_fit = self.X_pd.loc[:, selector.selected_columns]
        self.assertTrue(x.equals(x_from_fit))
