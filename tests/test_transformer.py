import unittest

from dwave.plugins.sklearn.transformers import HybridFeatureSelection

import numpy as np
import pandas as pd
import logging

class TestTransformer(unittest.TestCase):
    
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.rng = np.random.default_rng(1023884)
        self.X_np = None 
        self.y_np = None 
        self.X_pd = None 
        self.y_pd = None

    
    def test_init_good(self):
        """
        Enforcing defaults and making sure variables are set
        """
        a = HybridFeatureSelection()
        
        b = HybridFeatureSelection(alpha=.1)
        
        c = HybridFeatureSelection(alpha=.1,time_limit= 30)
        
        d = HybridFeatureSelection(time_limit=15)
        
        
        self.assertIsInstance(a, HybridFeatureSelection)
        self.assertIsInstance(b, HybridFeatureSelection)
        self.assertIsInstance(c, HybridFeatureSelection)
        self.assertIsInstance(d, HybridFeatureSelection)
        
        self.assertEqual(a.alpha, .5)
        self.assertEqual(b.alpha, .1)
        self.assertEqual(c.alpha, .1)
        self.assertEqual(d.alpha, .5)
        
        self.assertEqual(a.time_limit, 10)
        self.assertEqual(b.time_limit, 10)
        self.assertEqual(c.time_limit, 30)
        self.assertEqual(d.time_limit, 15)
        
        
        self.assertIsInstance(HybridFeatureSelection(alpha=0), HybridFeatureSelection)
        
    
    def test_init_bad(self):
        """
        Testing edges of initialization parameters
        """
        
        self.assertRaises(AssertionError, HybridFeatureSelection, alpha=-10)
        self.assertRaises(AssertionError, HybridFeatureSelection, alpha= 10)
        self.assertRaises(AssertionError, HybridFeatureSelection, time_limit= -100)
        self.assertRaises(AssertionError, HybridFeatureSelection, time_limit= 0)
        self.assertRaises(AssertionError, HybridFeatureSelection, time_limit= 1)
        
    def create_data_numpy(self) -> None:
        """
        Idempotent function that instantiates a class variable containing test numpy data
        """
        
        if self.X_np is None:
            self.X_np = self.rng.uniform(-10,10, size = (10000,300))
        
        if self.y_np is None:
            self.y_np = np.array([int(i) for i in (self.rng.uniform(0,1, size = (10000,1)) > .5)])

        return None 
    
    def create_data_pd(self) -> None:
        """
        Idempotent function that instantiates a class variable containing test pandas data
        derived from the numpy data. If `create_data_numpy` has not been called, this function 
        will call it. 
        """
        self.create_data_numpy()
        
        if self.X_pd is None:
            self.X_pd = pd.DataFrame(self.X_np)
        
        if self.y_pd is None:
            self.y_pd = pd.DataFrame(self.y_np)
        
    def test_fit_no_y(self): 
        """
        Test the fit method without an outcome variable specified
        """
        self.create_data_pd()
        
        selector = HybridFeatureSelection()
        
        # test default numpy
        
        selector.fit(self.X_np)
        self.assertEqual(len(selector.selected_columns), 10)
        
        try: 
            self.X_np[:,selector.selected_columns]
        except Exception as e:
            self.fail(e)
        
        # test default pandas 
        selector.fit(self.X_pd)
        self.assertEqual(len(selector.selected_columns), 10)
        
        try: 
            self.X_pd.loc[:,selector.selected_columns]
        except Exception as e:
            self.fail(e)
        
        # test non-default numpy
        
        selector.fit(self.X_np, number_of_features=30)
        self.assertEqual(len(selector.selected_columns), 30)
        
        try: 
            self.X_np[:,selector.selected_columns]
        except Exception as e:
            self.fail(e)
        
        # test non-default pandas 
        selector.fit(self.X_pd, number_of_features=30)
        self.assertEqual(len(selector.selected_columns),30)
        
        try: 
            self.X_pd.loc[:,selector.selected_columns]
        except Exception as e:
            self.fail(e)
            
        # test non-strict fit numpy 
        selector.fit(self.X_np, strict=False)
        self.assertLessEqual(len(selector.selected_columns), 10)
        
        try: 
            self.X_np[:,selector.selected_columns]
        except Exception as e:
            self.fail(e)
        
        # test non-strict fit pandas 
        selector.fit(self.X_pd, strict=False)
        self.assertLessEqual(len(selector.selected_columns), 10)
        
        try: 
            self.X_pd.loc[:,selector.selected_columns]
        except Exception as e:
            self.fail(e)
    
    def test_fit_y(self): 
        """
        Test the fit method with an outcome variable specified.
        """
        self.create_data_pd()
        selector = HybridFeatureSelection()
        
        # test default numpy
        
        selector.fit(self.X_np, self.y_np)
        self.assertEqual(len(selector.selected_columns), 10)
        
        try: 
            self.X_np[:,selector.selected_columns]
        except Exception as e:
            self.fail(e)
        
        # test default pandas 
        selector.fit(self.X_pd,  self.y_pd)
        self.assertEqual(len(selector.selected_columns), 10)
        
        try: 
            self.X_pd.loc[:,selector.selected_columns]
        except Exception as e:
            self.fail(e)
        
        # test non-default numpy
        
        selector.fit(self.X_np, self.y_np, number_of_features=30)
        self.assertEqual(len(selector.selected_columns), 30)

        try: 
            self.X_np[:,selector.selected_columns]
        except Exception as e:
            self.fail(e)
        
        # test non-default pandas 
        selector.fit(self.X_pd,  self.y_pd, number_of_features=30)
        self.assertEqual(len(selector.selected_columns),30)
        
        try: 
            self.X_pd.loc[:,selector.selected_columns]
        except Exception as e:
            self.fail(e)
            
        # test non-strict fit numpy 
        selector.fit(self.X_np,self.y_np, strict=False)
        self.assertLessEqual(len(selector.selected_columns), 10)
        
        try: 
            self.X_np[:,selector.selected_columns]
        except Exception as e:
            self.fail(e)
        
        # test non-strict fit pandas 
        selector.fit(self.X_pd, self.y_pd, strict=False)
        self.assertLessEqual(len(selector.selected_columns), 10)
        
        try: 
            self.X_pd.loc[:,selector.selected_columns]
        except Exception as e:
            self.fail(e)
        
    def test_transform_no_y(self): 
        """
        Test the transform method without the outcome variable specified
        """
        self.create_data_pd()
        selector = HybridFeatureSelection()
        
        # test numpy 
        selector.fit(self.X_np, number_of_features=20)
        
        x = selector.transform(self.X_np)
        
        self.assertEqual(x.shape[1], 20)
        
        x_from_fit = self.X_np[:,selector.selected_columns]
        self.assertTrue(np.array_equal(x, x_from_fit))
        
        # test pandas
        selector.fit(self.X_pd, number_of_features=20)
        
        x = selector.transform(self.X_pd)
        
        self.assertEqual(x.shape[1], 20)
        
        x_from_fit = self.X_pd.loc[:,selector.selected_columns]
        self.assertTrue(x.equals(x_from_fit))

    def test_transform_y(self): 
        """
        Test the transform method with the outcome variable specified
        """
        self.create_data_pd()
        selector = HybridFeatureSelection()
        
        # test numpy 
        selector.fit(self.X_np, self.y_np, number_of_features=20)
        
        x = selector.transform(self.X_np, self.y_np)
        
        self.assertEqual(x.shape[1], 20)
        
        x_from_fit = self.X_np[:,selector.selected_columns]
        self.assertTrue(np.array_equal(x, x_from_fit))
        
        # test pandas
        selector.fit(self.X_pd, self.y_pd, number_of_features=20)
        
        x = selector.transform(self.X_pd, self.y_pd)
        
        self.assertEqual(x.shape[1], 20)
        
        x_from_fit = self.X_pd.loc[:,selector.selected_columns]
        self.assertTrue(x.equals(x_from_fit))
        
        # test numpy without fit
        selector.unfit()
        
        x = selector.transform(self.X_np, self.y_np, number_of_features=20)
        
        self.assertEqual(x.shape[1], 20)
        
        x_from_fit = self.X_np[:,selector.selected_columns]
        self.assertTrue(np.array_equal(x, x_from_fit))
        
        # test pandas withoput fit
        selector.unfit()
        
        x = selector.transform(self.X_pd, self.y_pd, number_of_features=20)
        
        self.assertEqual(x.shape[1], 20)
        
        x_from_fit = self.X_pd.loc[:,selector.selected_columns]
        self.assertTrue(x.equals(x_from_fit))
        
    def test_fit_transform_no_y(self): 
        """
        Test the transform method without the outcome variable specified
        """
        self.create_data_pd()
        selector = HybridFeatureSelection()
        
        # test numpy without fit
        x = selector.fit_transform(self.X_np, number_of_features=20)
        
        self.assertEqual(x.shape[1], 20)
        
        x_from_fit = self.X_np[:,selector.selected_columns]
        
        self.assertTrue(np.array_equal(x, x_from_fit))
        
        # test pandas withoput fit
        x = selector.fit_transform(self.X_pd, number_of_features=20)
        
        self.assertEqual(x.shape[1], 20)
        
        x_from_fit = self.X_pd.loc[:,selector.selected_columns]
        self.assertTrue(x.equals(x_from_fit))

    def test_fit_transform_y(self): 
        """
        Test the transform method with the outcome variable specified
        """
        self.create_data_pd()
        selector = HybridFeatureSelection()
        
        # test numpy without fit
        x = selector.fit_transform(self.X_np, self.y_np, number_of_features=20)
        
        self.assertEqual(x.shape[1], 20)
        
        x_from_fit = self.X_np[:,selector.selected_columns]
        self.assertTrue(np.array_equal(x, x_from_fit))
        
        # test pandas withoput fit
        x = selector.fit_transform(self.X_pd, self.y_pd, number_of_features=20)
        
        self.assertEqual(x.shape[1], 20)
        
        x_from_fit = self.X_pd.loc[:,selector.selected_columns]
        self.assertTrue(x.equals(x_from_fit))
    
    def test_unfit(self):
        """
        Test `unfit` function
        """
        self.create_data_pd()
        selector = HybridFeatureSelection()
        
        selector.fit(self.X_np, self.y_np)
        
        selector.unfit()
        
        self.assertIsNone(selector.selected_columns)
    
    def test_update_time_limit(self):
        """
        Test the `update_time_limit` function.
        """
        selector = HybridFeatureSelection()
        selector.update_time_limit(100)
        self.assertEqual(selector.time_limit, 100)
        

class TestManyFeatures(unittest.TestCase):
    
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.rng = np.random.default_rng(1023884)
        self.X_np = None 
        self.y_np = None 
        self.X_pd = None 
        self.y_pd = None
        
    def create_data_numpy(self) -> None:
        """
        Idempotent function that instantiates a class variable containing test numpy data
        """
        
        if self.X_np is None:
            self.X_np = self.rng.uniform(-10,10, size = (10000,3000))
        
        if self.y_np is None:
            self.y_np = np.array([int(i) for i in (self.rng.uniform(0,1, size = (10000,1)) > .5)])

        return None 
    
    def create_data_pd(self) -> None:
        """
        Idempotent function that instantiates a class variable containing test pandas data
        derived from the numpy data. If `create_data_numpy` has not been called, this function 
        will call it. 
        """
        self.create_data_numpy()
        
        if self.X_pd is None:
            self.X_pd = pd.DataFrame(self.X_np)
        
        if self.y_pd is None:
            self.y_pd = pd.DataFrame(self.y_np)
    
    def test_fit_transform_no_y(self): 
        """
        Test the transform method without the outcome variable specified
        """
        self.create_data_pd()
        selector = HybridFeatureSelection(time_limit = 30)
        
        # test numpy without fit
        x = selector.fit_transform(self.X_np, number_of_features=20)
        
        self.assertEqual(x.shape[1], 20)
        
        x_from_fit = self.X_np[:,selector.selected_columns]
        
        self.assertTrue(np.array_equal(x, x_from_fit))
        
        # test pandas withoput fit
        x = selector.fit_transform(self.X_pd, number_of_features=20)
        
        self.assertEqual(x.shape[1], 20)
        
        x_from_fit = self.X_pd.loc[:,selector.selected_columns]
        self.assertTrue(x.equals(x_from_fit))

    def test_fit_transform_y(self): 
        """
        Test the transform method with the outcome variable specified
        """
        self.create_data_pd()
        selector = HybridFeatureSelection(time_limit = 30)
        
        # test numpy without fit
        x = selector.fit_transform(self.X_np, self.y_np, number_of_features=20)
        
        self.assertEqual(x.shape[1], 20)
        
        x_from_fit = self.X_np[:,selector.selected_columns]
        self.assertTrue(np.array_equal(x, x_from_fit))
        
        # test pandas withoput fit
        x = selector.fit_transform(self.X_pd, self.y_pd, number_of_features=20)
        
        self.assertEqual(x.shape[1], 20)
        
        x_from_fit = self.X_pd.loc[:,selector.selected_columns]
        self.assertTrue(x.equals(x_from_fit))


class TestVeryManyFeatures(unittest.TestCase):
    
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.rng = np.random.default_rng(1023884)
        self.X_np = None 
        self.y_np = None 
        self.X_pd = None 
        self.y_pd = None
        
    def create_data_numpy(self) -> None:
        """
        Idempotent function that instantiates a class variable containing test numpy data
        """
        
        if self.X_np is None:
            self.X_np = self.rng.uniform(-10,10, size = (10000,25000))
        
        if self.y_np is None:
            self.y_np = np.array([int(i) for i in (self.rng.uniform(0,1, size = (10000,1)) > .5)])

        return None 
    
    def create_data_pd(self) -> None:
        """
        Idempotent function that instantiates a class variable containing test pandas data
        derived from the numpy data. If `create_data_numpy` has not been called, this function 
        will call it. 
        """
        self.create_data_numpy()
        
        if self.X_pd is None:
            self.X_pd = pd.DataFrame(self.X_np)
        
        if self.y_pd is None:
            self.y_pd = pd.DataFrame(self.y_np)
    
    def test_fit_transform_no_y(self): 
        """
        Test the transform method without the outcome variable specified
        """
        logging.getLogger().setLevel(logging.INFO)
        
        self.create_data_pd()
        selector = HybridFeatureSelection(time_limit = 30)
        
        # test numpy without fit
        x = selector.fit_transform(self.X_np, number_of_features=20)
        
        self.assertEqual(x.shape[1], 20)
        
        x_from_fit = self.X_np[:,selector.selected_columns]
        
        self.assertTrue(np.array_equal(x, x_from_fit))
        
        # test pandas withoput fit
        x = selector.fit_transform(self.X_pd, number_of_features=20)
        
        self.assertEqual(x.shape[1], 20)
        
        x_from_fit = self.X_pd.loc[:,selector.selected_columns]
        self.assertTrue(x.equals(x_from_fit))

    def test_fit_transform_y(self): 
        """
        Test the transform method with the outcome variable specified
        """
        logging.getLogger().setLevel(logging.INFO)
        
        self.create_data_pd()
        selector = HybridFeatureSelection(time_limit = 30)
        
        # test numpy without fit
        x = selector.fit_transform(self.X_np, self.y_np, number_of_features=20)
        
        self.assertEqual(x.shape[1], 20)
        
        x_from_fit = self.X_np[:,selector.selected_columns]
        self.assertTrue(np.array_equal(x, x_from_fit))
        
        # test pandas withoput fit
        x = selector.fit_transform(self.X_pd, self.y_pd, number_of_features=20)
        
        self.assertEqual(x.shape[1], 20)
        
        x_from_fit = self.X_pd.loc[:,selector.selected_columns]
        self.assertTrue(x.equals(x_from_fit))
        
if __name__ == '__main__':
    unittest.main()