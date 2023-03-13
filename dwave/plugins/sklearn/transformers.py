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
from dwave.cloud.exceptions import ConfigFileError, SolverAuthenticationError
from typing import Union
from .utilities import corrcoef

__all__ = ["SelectFromQuadraticModel"]


class SelectFromQuadraticModel(BaseEstimator, SelectorMixin):
    """scikit-learn `SelectorMizin` which uses the `LeapHybridCQMSampler` solver."""

    acceptable_methods = ["correlation", "mutual information"]

    def __init__(
        self,
        alpha: float = 0.5,
        time_limit: int = 10,
        n_default_feature: int = 10,
        method: str = "correlation",
        chunksize=None,
    ) -> None:
        """Instantiate `SelectFromQuadraticModel`

        Args:
            alpha: Weighting hyperparameter for the objective . Must be between 0 and 1.
            time_limit: Runtime limit for the Leap hybrid CQM solver.
            n_default_feature: Number of features to select. Ignored if ``number_of_features`` 
                is configured in the ``fit`` or ``fit_transform`` methods. 
            method: Method of formulating the feature selection problem. Only "correlation" is supported. 
            chunksize: Used for testing internal memory usage. 
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

    def _get_support_mask(self) -> list:
        """Get the boolean mask indicating which features are selected

        Args:

        Returns:
          boolean array of shape [# input features]: An element is True iff its corresponding feature is selected for
          retention.Returns

        Raises:
            RuntimeError: This method will raise an error if it is run before `fit`
        """

        if self.mask is None:
            raise RuntimeError("fit hasn't been run yet")

        return self.mask

    def calculate_correlation_matrix(
        self, X: np.ndarray, y: Union[np.ndarray, None] = None
    ) -> np.ndarray:
        """Calculate the correlation matrix. 
        
            Calculates correlations between the feature columns. Optionally calculates correlations with 
            outcome for the matrix diagonals.

        Args:
            X:  Features as an array of shape ``(n_observations, n_features)``.
            y: Outcome variables as an array of shape ``(n_observations, 1)``.

        Returns:
            np.ndarray:  Correlation matrix of shape ``(n_features, n_features)``. The diagonal is 1 
            if no ``y`` is given; otherwise it is ``corr(x_i, y)``.
        """
        # generate correlation matrix, use tempfile and chunked process

        f_x = tempfile.NamedTemporaryFile()
        f_corr = tempfile.NamedTemporaryFile()

        X_memmapped = np.memmap(f_x, "float64", mode="w+", shape=X.shape)
        X_memmapped[:] = X[:]
        X_memmapped.flush()

        correlation_matrix = np.memmap(
            f_corr, "float64", mode="w+", shape=(X.shape[1], X.shape[1])
        )
        corrcoef(X_memmapped, out=correlation_matrix, copy=False, rowvar=False)

        if y is None:
            np.fill_diagonal(
                correlation_matrix, [0 for _ in range(correlation_matrix.shape[1])]
            )
        else:
            X_memmapped = (X_memmapped - X_memmapped.mean(axis=0)) / (
                X_memmapped.std(axis=0)
            )
            y_standardized = (y - y.mean()) / y.std()
            np.fill_diagonal(
                correlation_matrix,
                self.alpha * (1 / y.shape[0]) * (X_memmapped.T @ y_standardized),
            )

        f_x.close()

        return correlation_matrix, f_corr

    def calculate_mutual_information(self, X: np.ndarray, y: np.ndarray):
        """Calculate mutual information matrix.

        Args:
            X: Features as an array of shape ``(n_observations, n_features)``.
            y: Outcome variables as an array of shape ``(n_observations, 1)``.

        Raises:
            NotImplementedError: Always raised (this function is not currently implemented).
        """
        raise NotImplementedError("Mutual information is not yet implemented")

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.DataFrame, pd.Series, None] = None,
        number_of_features: int = None,
        strict: bool = True,
    ):
        """Select the features to keep.

        Args:
            X: Features as a matrix-like object where columns are the features to be selected 
                and rows are observations. If mutual information is selected, non-binary 
                rows are assumed to be continuous.
            y: Outcome variables to be incorporated in the correlation matrix along with 
                the covariances among features. Required for mutual information. 
            number_of_features: Number of features to be selected. If ``strict`` is ``True``, 
                exactly this number of features is selected; otherwise this is an upper bound. 
                If set to ``None``, ``n_default_feature`` features are selected.
            strict (bool, optional): If `True` exactly `number_of_features` is selected, otherwise at most `number_of_features` is selected. Defaults to True.

        Returns:
            SelectFromQuadraticModel: This instance of `SelectFromQuadraticModel`
        """
        if number_of_features is None:
            number_of_features = self.n_default_feature

        if hasattr(X, "toarray") and (not isinstance(X, pd.DataFrame)):
            X = X.toarray()

        if self.method == "correlation":
            model_matrix, f_model = self.calculate_correlation_matrix(X, y)
        elif self.method == "mutual information":
            if y is None:
                raise ValueError("mutual infromation requires outcome")
            model_matrix, f_model = self.calculate_mutual_information(X, y)
        else:
            raise ValueError(f"Only methods {self.acceptable_methods} are implimented")

        # feature selection formulation (same as example)

        logging.info(f"constructing feature selection model")

        if self.chunksize is not None:
            chunksize = self.chunksize
        else: 
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

        f_model.close()

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

        try:
            cqm_solver = LeapHybridCQMSampler()
        except (ConfigFileError, SolverAuthenticationError) as e:
            raise RuntimeError(
                f"""The Leap hybrid solver raised the following error: {e}. 
                
                There is a high likelihood that dwave.system is not set up properly or you are missing a Leap token. 
                If you did not configure dwave.system please see the installation guide
                    
                    https://docs.ocean.dwavesys.com/en/stable/overview/install.html
                
                If you have installed dwave.system but need more details on configuration see 

                    https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html
                """
            )

        min_time_limit = cqm_solver.min_time_limit(feature_selection_cqm)

        if self.time_limit < min_time_limit:
            raise ValueError(
                f"the time limit must be at least for this problem {min_time_limit}"
            )

        feature_sample: dimod.SampleSet = cqm_solver.sample_cqm(
            feature_selection_cqm, time_limit=self.time_limit
        )

        logging.info("CQM sampling done")

        # use sample to get selected features

        feature_sample_feasible = feature_sample.filter(lambda d: d.is_feasible)

        if len(feature_sample_feasible) != 0:
            feature_sample_best = feature_sample_feasible.first
        else:
            feature_sample_best = feature_sample.first
            warnings.warn(
                "No feasible selection found, using lowest energy", RuntimeWarning
            )
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
        """Returns X selected for the features decided in `self.fit()`. If `self.fit()` has not been called yet then transform will first call it.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): a matrix-like object where each row is an observation, and each column is a feature to be selected.
            y (Union[np.ndarray, pd.DataFrame, pd.Series, None], optional): Outcome to be considered in feature selection. If provided, the correlation with the outcome will be considered,
            if not then only the covariances among the features will be considered. Only used if `fit()` has not been called already. Defaults to None.

        Returns:
            (Union[np.ndarray, pd.DataFrame]): Same type as X, with the selected subset of features
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
        """undoes the `fit()` method"""
        self.selected_columns = None
        self.mask = None
        return None

    def update_time_limit(self, time_limit: int) -> None:
        """update the time limit"""
        self.time_limit = time_limit
        return None
