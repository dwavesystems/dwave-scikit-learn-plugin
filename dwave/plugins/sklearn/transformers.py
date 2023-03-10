# Copyright 2023 D-Wave Systems Inc.
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

import tempfile
import logging
import warnings

import dimod
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from dwave.system import LeapHybridCQMSampler
from typing import Union

__all__ = ["SelectFromQuadraticModel"]


class SelectFromQuadraticModel(BaseEstimator, SelectorMixin):
    """_summary_"""

    acceptable_methods = ["correlation", "mutual information"]

    def __init__(
        self,
        alpha: float = 0.5,
        time_limit: int = 10,
        n_default_feature: int = 10,
        method: str = "correlation",
        chunksize = None
    ) -> None:
        """
        Initialize a hybrid quantum-classical feature selection transformer.

        Args:
            alpha (float, optional): hyper-parameter which weights correlation to outcome vs covariance. Must be between 0 and 1. Defaults to .5.
            time_limit (int, optional): a time limit for the Leap Constrained Quadratic Model Solver to solve the feature selection problem.
            Must be at least the minimum time limit of the solver. Defaults to 10.
            n_default_features (int, optional)
            method (string, optional): The method used to calculate the model matrix. One of "correlation" or "mutual information". Defaults to correlation.
        """
        super().__init__()
        self.alpha = alpha
        self.time_limit = time_limit
        self.n_default_feature = n_default_feature
        self.selected_columns = None
        self.mask = None
        if (self.alpha > 1) or (self.alpha < 0):
            raise ValueError(f"alpha {self.alpha} is not between 0 and 1")

        if self.time_limit <= 1:
            raise ValueError("Time limit must be positive and greater than 1")

        if method not in self.acceptable_methods:
            raise ValueError(
                f"method was {method}, must be one of {SelectFromQuadraticModel.acceptable_methods}"
            )

        self.method = method
        
        self.chunksize = chunksize

    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected
        Returns
        -------
        support : boolean array of shape [# input features]
            An element is True iff its corresponding feature is selected for
            retention.
        """
        if self.mask is None:
            raise RuntimeError("fit hasn't been run yet")

        return self.mask

    def calculate_correlation_matrix(
        self, X: np.ndarray, y: Union[np.ndarray, None] = None
    ):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.
            chunksize (int, optional): _description_. Defaults to 500.

        Returns:
            _type_: _description_
        """
        logging.info(
            "Starting correlation calculation"
        )  
        if self.chunksize is None:
            chunksize = 500
        else: 
            chunksize = self.chunksize
        
        # heavy logging to diagnose memory performance
        # generate correlation matrix, use tempfile and chunked process if too big
        if X.shape[1] < chunksize:
            logging.info("Using numpy corrcoef")
            correlation_matrix = np.corrcoef(X, rowvar=False)
            X_standardized = None
        else:
            logging.info("data is too large, will be chunked")

            with tempfile.NamedTemporaryFile() as f:
                X_base = np.memmap(f, "float64", mode="w+", shape=X.shape)

            if isinstance(X, pd.DataFrame):
                X_base[:] = X.to_numpy()
            else:
                X_base[:] = X[:]

            X_base.flush()

            logging.info("standardizing features")
            with tempfile.NamedTemporaryFile() as f:
                X_standardized = np.memmap(f, "float64", mode="w+", shape=X.shape)

            X_standardized = (X_base - X_base.mean(axis=0)) / X_base.std(axis=0)

            n_features = X_standardized.shape[1]

            with tempfile.NamedTemporaryFile() as temp:
                correlation_matrix = np.memmap(
                    temp, "float64", mode="w+", shape=(n_features, n_features)
                )

            logging.info("calculating chunked correlations")
            for row_index in range(0, n_features, chunksize):
                for col_index in range(0, n_features, chunksize):
                    row_index_max = min(row_index + chunksize, n_features)
                    col_index_max = min(col_index + chunksize, n_features)
                    chunk1 = X_standardized[:, row_index:row_index_max]
                    chunk2 = X_standardized[:, col_index:col_index_max]
                    correlation_matrix[
                        row_index:row_index_max, col_index:col_index_max
                    ] = (chunk1.T @ chunk2)
                    correlation_matrix.flush()
            correlation_matrix = 1 / X_standardized.shape[0] * correlation_matrix

        logging.info("correlation matrix calculated")
        # sub in diagonal for correlation with outcome (if there is an outcome)
        if y is not None:
            logging.info("calculating outcome correlation")
            if X_standardized is None:
                X_standardized = (X - X.mean(axis=0)) / X.std(axis=0)

            y_standardized = (y - y.mean()) / y.std()

            np.fill_diagonal(
                correlation_matrix,
                self.alpha * (1 / y.shape[0]) * (X_standardized.T @ y_standardized),
            )
        else:
            np.fill_diagonal(correlation_matrix, [0 for _ in range(X.shape[1])])

        return correlation_matrix

    def calculate_mutual_information(
        self, X: np.ndarray, y: Union[np.ndarray, None] = None
    ):
        raise NotImplementedError("Mutual information is not yet implemented")

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame, pd.Series, None] = None,
        number_of_features: int = None,
        strict: bool = True
        ):
        """
        Conducts the feature selection proceedure and finds the features to keep.
        Args:
            X (_type_): a matrix-like object where each row is an observation, and each column is a feature to be selected. If mutual information is selected
                        every non-binary row will be assumed to be continuous.
            y (_type_, optional): optional outcome to be considered in feature selection. If provided, the correlation with the outcome will be considered,
                                  if not then only the covariances among the features will be considered. Required for mutual information. Defaults to None.
            number_of_features (int, optional): The number of features to be selected. If `strict` is `True` exactly this number of features,
                                                otherwise at most this number of features is selected. Defaults to number given at construction.
            strict (bool, optional): If `True` exactly `number_of_features` is selected, otherwise at most `number_of_features` is selected. Defaults to True.
        """
        if number_of_features is None:
            number_of_features = self.n_default_feature
        
        if hasattr(X, "toarray") and (not isinstance(X, pd.DataFrame)):
            X = X.toarray()

        if self.method == "correlation":
                model_matrix = self.calculate_correlation_matrix(X, y)
        elif self.method == "mutual information":
            if y is None:
                raise ValueError("mutual infromation requires outcome")
            model_matrix = self.calculate_mutual_information(X, y)
        else:
            raise ValueError(f"Only methods {self.acceptable_methods} are implimented")

        # feature selection formulation (same as example)

        logging.info(f"constructing feature selection model")

        chunksize = 100

        if X.shape[1] < chunksize:
            logging.info("constructing BQM using dimod constructor from matrix")

            feature_selection_bqm = dimod.BQM(model_matrix, "BINARY")
        else:
            logging.info("constructing bqm using chunked iteration")

            feature_selection_bqm = dimod.BQM(vartype="BINARY")
            feature_selection_bqm.add_variables_from(
                {i: model_matrix[i, i] for i in range(model_matrix.shape[0])}
            )

            logging.info("variables and linear biases added")

            for row_index in range(0, model_matrix.shape[0], chunksize):
                for col_index in range(0, model_matrix.shape[1], chunksize):
                    row_index_max = min(row_index + chunksize, model_matrix.shape[0])
                    col_index_max = min(col_index + chunksize, model_matrix.shape[0])
                    chunk = model_matrix[
                        row_index:row_index_max, col_index:col_index_max
                    ]
                    quad_biases = [
                        (i[0], i[1], j) for i, j in np.ndenumerate(chunk) if i[0] > i[1]
                    ]
                    feature_selection_bqm.add_interactions_from(quad_biases)

        logging.info("BQM created")

        feature_selection_cqm = dimod.CQM()
        feature_selection_cqm.set_objective(feature_selection_bqm)
        logging.info("CQM created (objective only)")
        
        sense = "==" if strict else "<="
        
        feature_selection_cqm.add_constraint_from_iterable(
            [(var, 1) for var in feature_selection_cqm.variables],
            sense,
            number_of_features,
        )
              
        cqm_solver = LeapHybridCQMSampler()

        min_time_limit = cqm_solver.min_time_limit(feature_selection_cqm)
        
        if self.time_limit < min_time_limit:
            raise ValueError(
                f"the time limit must be at least for this problem {min_time_limit}"
            )
        
        feature_sample : dimod.SampleSet = cqm_solver.sample_cqm(feature_selection_cqm, time_limit = self.time_limit)

        logging.info("CQM sampling done")

        # use sample to get selected features
        
        feature_sample_feasible = feature_sample.filter(lambda d: d.is_feasible)
        
        if len(feature_sample_feasible) != 0: 
            feature_sample_best = feature_sample_feasible.first
        else:
            feature_sample_best = feature_sample.first
            warnings.warn("No feasible selection found, using lowest energy", RuntimeWarning)
            raise RuntimeError()
        selected_features = [
            index for index, val in feature_sample_best.sample.items() if val == 1
        ]
        
        if isinstance(X, pd.DataFrame):
            self.selected_columns = X.columns[selected_features]
            self.mask = np.array([(col in self.selected_columns) for col in X.columns])
        else:
            self.selected_columns = selected_features
            self.mask = np.array(
                [(i in self.selected_columns) for i in range(X.shape[1])]
            )

        return self

    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame, pd.Series, None] = None,
        **kwargs,
    ):
        """
        Returns X selected for the features decided in `self.fit()`. If `self.fit()` has not been called yet then transform will first call it.

        Args:
            X (_type_): a matrix-like object where each row is an observation, and each column is a feature to be selected.
            y (_type_, optional): optional outcome to be considered in feature selection. If provided, the correlation with the outcome will be considered,
                                  if not then only the covariances among the features will be considered. Only used if `fit()` has not been called already. Defaults to None..
            kwargs (_type_, optional): optionally passed to the fit method if not yet called.

        Returns:
            X (_type_): X with the selected subset of features
        """
        if self.mask is None:
            self.fit(X, y, **kwargs)

        if isinstance(X, pd.DataFrame):
            X = X.loc[:, self.selected_columns]
        elif isinstance(X, np.ndarray):
            X = X[:, self.selected_columns]
        else:
            X = super().transform(X)

        return X

    def unfit(self) -> None:
        """
        undoes the `fit()` method
        """
        self.selected_columns = None
        self.mask = None
        return None

    def update_time_limit(self, time_limit: int) -> None:
        """
        update the time limit
        """
        self.time_limit = time_limit
        return None
