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

from __future__ import annotations

import itertools
import logging
import tempfile
import typing
import warnings

import numpy as np
import numpy.typing as npt

from dwave.cloud.exceptions import ConfigFileError, SolverAuthenticationError
from dwave.system import LeapHybridNLSampler
from dwave.optimization import Model

from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

from dwave.plugins.sklearn.utilities import corrcoef

__all__ = ["SelectFromNonlinearModel"]


class SelectFromNonlinearModel(SelectorMixin, BaseEstimator):
    """Select features using a quadratic optimization problem solved on a hybrid solver.

    Args:
        alpha:
            Hyperparameter between 0 and 1 that controls the relative weight of
            the relevance and redundancy terms.
            ``alpha=0`` places no weight on the quality of the features,
            therefore the features will be selected as to minimize the
            redundancy without any consideration to quality.
            ``alpha=1`` places the maximum weight on the quality of the features,
            and therefore will be equivalent to using
            :class:`sklearn.feature_selection.SelectKBest`.
        num_features:
            The number of features to select.
        time_limit:
            The time limit for the run on the hybrid solver.

    """

    ACCEPTED_METHODS = [
        "correlation",
        # "mutual information",  # todo
        ]

    def __init__(
        self,
        *,
        alpha: float = .5,
        method: str = "correlation",  # undocumented until there is another supported
        num_features: int = 10,
        time_limit: typing.Optional[float] = None,
    ):
        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be between 0 and 1, given {alpha}")

        if method not in self.ACCEPTED_METHODS:
            raise ValueError(
                f"method must be one of {self.ACCEPTED_METHODS}, given {method}"
            )

        if num_features <= 0:
            raise ValueError(f"num_features must be a positive integer, given {num_features}")

        self.alpha = alpha
        self.method = method
        self.num_features = num_features
        self.time_limit = time_limit  # check this lazily

    def __sklearn_is_fitted__(self) -> bool:
        # used by `check_is_fitted()`
        try:
            self._mask
        except AttributeError:
            return False

        return True

    def _get_support_mask(self) -> np.ndarray[typing.Any, np.dtype[np.bool_]]:
        """Get the boolean mask indicating which features are selected

        Returns:
          boolean array of shape [# input features]. An element is True iff its
          corresponding feature is selected for retention.

        Raises:
            RuntimeError: This method will raise an error if it is run before `fit`
        """
        check_is_fitted(self)

        try:
            return self._mask
        except AttributeError:
            raise RuntimeError("fit hasn't been run yet")

    @staticmethod
    def correlation_nl(
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        *,
        alpha: float,
        num_features: int,
        strict: bool = True,
    ) -> tuple[Model, Model.binary(), np.array()]: 
        """Build a nonlinear model for feature selection.

        This method is based on maximizing influence and feature independence as
        measured by correlation [Milne et al.]_.

        Args:
            X:
                Feature vectors formatted as a numerical 2D array-like.
            y:
                Class labels formatted as a numerical 1D array-like.
            alpha:
                Hyperparameter between 0 and 1 that controls the relative weight of
                the relevance and redundancy terms.
                ``alpha=0`` places no weight on the quality of the features,
                therefore the features will be selected as to minimize the
                redundancy without any consideration to quality.
                ``alpha=1`` places the maximum weight on the quality of the features,
                and therefore will be equivalent to using
                :class:`sklearn.feature_selection.SelectKBest`.
            num_features:
                The number of features to select.
            strict:
                If ``False`` the constraint on the number of selected features
                is ``<=`` rather than ``==``.

        Returns:
            A nonlinear model and the binary list.

        .. [Milne et al.] Milne, Andrew, Maxwell Rounds, and Phil Goddard. 2017. "Optimal Feature
            Selection in Credit Scoring and Classification Using a Quantum Annealer."
            1QBit; White Paper.
            https://1qbit.com/whitepaper/optimal-feature-selection-in-credit-scoring-classification-using-quantum-annealer
        """

        X = np.atleast_2d(np.asarray(X))
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array-like")

        if y.ndim != 1:
            raise ValueError("y must be a 1-dimensional array-like")

        if y.shape[0] != X.shape[0]:
            raise ValueError(f"requires: X.shape[0] == y.shape[0] but {X.shape[0]} != {y.shape[0]}")

        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be between 0 and 1, given {alpha}")

        if num_features <= 0:
            raise ValueError(f"num_features must be a positive integer, given {num_features}")

        if X.shape[0] <= 1:
            raise ValueError("X must have at least two rows")

        

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

            # our objective

            # Note: the full symmetric matrix (with both upper- and lower-diagonal
            # entries for each correlation coefficient) is retained for consistency with
            # the original formulation from Milne et al.
            # initialize model, create binary list, make constant
            NL = Model()
            total_num_features=X.shape[1]
            
            
            X_binary = NL.binary(total_num_features) 
            var_features = NL.constant(num_features)
            feat_corr = correlations[:-1,:-1]
            
            # take last element in every row
            label_corr = np.array(correlations[:-1,-1])

            # Make a constant node in order to splice and use in objective
            NL_corr = NL.constant(feat_corr)

            # extract upper triangle, excluding diagonal. Flatten into 1D array
            C = np.triu(NL_corr, k=1).flatten()

            # generate all column and row indices
            quad_col = np.tile(np.arange(total_num_features), total_num_features)
            quad_row = np.tile(np.arange(total_num_features), 
                        (total_num_features,1)).flatten('F')

            # extract indices where correlation value not equal to zero
            
            # j index
            q2 = quad_col[C != 0]
            # i index
            q1 = quad_row[C != 0]

            # extract values at position (i,j) where not equal to zero
            q3 = C[C != 0]

            # 1D numpy array initialized to size of num_rows with 0 in every position
            linear = np.zeros(len(feat_corr[0]))
            expected_linear = np.zeros(len(feat_corr[0]))

            # numpy will automatically go element-by-element in the arrays
            linear += NL.constant(-1.0 * label_corr * alpha * num_features)
            expected_linear += (-1.0 * label_corr * alpha * num_features)

            # if must choose exact number of desired features
            if strict:
                NL.add_constraint(X_binary.sum() == var_features)
            else:
                NL.add_constraint(X_binary.sum() <= var_features)

            NL.minimize(NL.constant(2.0) * NL.quadratic_model(X_binary, quadratic=(q3, [q1, q2]), linear=linear))
            
        return NL, X_binary, expected_linear

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        *,
        alpha: typing.Optional[float] = None,
        num_features: typing.Optional[int] = None,
        time_limit: typing.Optional[float] = None,
    ) -> SelectFromNonlinearModel:
        """Select the features to keep.

        Args:
            X:
                Feature vectors formatted as a numerical 2D array-like.
            y:
                Class labels formatted as a numerical 1D array-like.
            alpha:
                Hyperparameter between 0 and 1 that controls the relative weight of
                the relevance and redundancy terms.
                ``alpha=0`` places no weight on the quality of the features,
                therefore the features will be selected as to minimize the
                redundancy without any consideration to quality.
                ``alpha=1`` places the maximum weight on the quality of the features,
                and therefore will be equivalent to using
                :class:`sklearn.feature_selection.SelectKBest`.
            num_features:
                The number of features to select.
                Defaults to the value provided to the constructor.
            time_limit:
                The time limit for the run on the hybrid solver.
                Defaults to the value provided to the constructor.

        Returns:
            This instance of `SelectFromNonlinearModel`.
        """
        X = np.atleast_2d(np.asarray(X))
        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array-like")

        # y is checked by the correlation method function

        if alpha is None:
            alpha = self.alpha
        # alpha is checked by the correlation method function

        if num_features is None:
            num_features = self.num_features
        # num_features is checked by the correlation method function

        # if we already have fewer features than requested, just return
        if num_features >= X.shape[1]:
            self._mask = np.ones(X.shape[1], dtype=bool)
            return self
        
        if self.method == "correlation":
            NL, X_binary, _ = self.correlation_nl(X, y, num_features=num_features, alpha=alpha)

        else:
            raise ValueError(f"only methods {self.acceptable_methods} are implemented")

        try:
            sampler = LeapHybridNLSampler()
        except (ConfigFileError, SolverAuthenticationError) as e:
            raise RuntimeError(
                f"""Instantiation of a Leap hybrid solver failed with an {e} error.

                See https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html for configuring
                access to Leap’s solvers.
                """
            )

        # time_limit is checked by the LeapHybridNLSampler
        sampler.sample(NL, time_limit=self.time_limit, label='NL Plug-IN')
        
        # Get the index position of chosen features
        # Example Given (e.g.) of 6 features to choose 3
        with NL.lock():
            selected = X_binary.state(0) # e.g. [0,1,0,0,1,1,0]
            NL.unlock()

        mask = np.asarray(selected, dtype=bool) # e.g. [False, True, False, False, True, True, False]
        self._mask = mask

        return self

    def unfit(self):
        """Undo a previously executed ``fit`` method."""
        del self._mask