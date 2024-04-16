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

from itertools import combinations
import tempfile
import typing

import dimod
from dwave.cloud.exceptions import ConfigFileError, SolverAuthenticationError
from dwave.plugins.sklearn.utilities import (
    corrcoef,
    _compute_cmi_distance,
    _compute_mi,
    _iterate_columns)
from dwave.system import LeapHybridCQMSampler
import numpy as np
import numpy.typing as npt
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y

__all__ = ["SelectFromQuadraticModel"]


class SelectFromQuadraticModel(SelectorMixin, BaseEstimator):
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
        method:
            If equal to ``correlation`` uses a correlation as a criterion for
            choosing the features according to [1]_. If equal to ``mutual_information``
            uses mutual information criteria [2]_ or [3]_ (advances feature).
        time_limit:
            The time limit for the run on the hybrid solver.

    .. [1] [Milne et al.] Milne, Andrew, Maxwell Rounds, and Phil Goddard. 2017. "Optimal Feature
            Selection in Credit Scoring and Classification Using a Quantum Annealer."
            1QBit; White Paper.
            https://1qbit.com/whitepaper/optimal-feature-selection-in-credit-scoring-classification-using-quantum-annealer
    .. [2] Peng, F. Long, and C. Ding. Feature selection based on mutual information criteria
            of max-dependency, max-relevance, and min-redundancy. IEEE Transactions on pattern
            analysis and machine intelligence, 27(8):1226–1238, 2005.
    .. [3] X. V. Nguyen, J. Chan, S. Romano, and J. Bailey. Effective global approaches for
            mutual information based feature selection. In Proceedings of the 20th ACM SIGKDD
            international conference on Knowledge discovery and data mining,
            pages 512–521. ACM, 2014.
    """

    ACCEPTED_METHODS = [
        "correlation",
        "mutual_information",
    ]

    def __init__(
        self,
        *,
        alpha: float = .5,
        method: str = "correlation",
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

    def correlation_cqm(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        *,
        alpha: float,
        num_features: int,
        strict: bool = True,
    ) -> dimod.ConstrainedQuadraticModel:
        """Build a constrained quadratic model for feature selection.

        This method is based on maximizing influence and independence as
        measured by correlation [1]_.

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
            A constrained quadratic model.

        .. [1] Milne, Andrew, Maxwell Rounds, and Phil Goddard. 2017. "Optimal Feature
            Selection in Credit Scoring and Classification Using a Quantum Annealer."
            1QBit; White Paper.
            https://1qbit.com/whitepaper/optimal-feature-selection-in-credit-scoring-classification-using-quantum-annealer
        """
        self._check_params(X, y, alpha, num_features)

        cqm = dimod.ConstrainedQuadraticModel()
        cqm.add_variables(dimod.BINARY, X.shape[1])

        # add the k-hot constraint
        cqm.add_constraint(
            ((v, 1) for v in cqm.variables),
            '==' if strict else '<=',
            num_features,
            label=f"{num_features}-hot",
        )

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
            # multiplying all but last columns and rows with (1 - alpha)
            np.multiply(correlations[:-1, :-1], (1 - alpha), out=correlations[:-1, :-1])
            # our objective
            # we multiply by num_features to have consistent performance
            # with the increase of the number of features
            np.fill_diagonal(correlations, correlations[:, -1] * (- alpha * num_features))
            # Note: we only add terms on and above the diagonal
            it = np.nditer(correlations[:-1, :-1], flags=['multi_index'], op_flags=[['readonly']])
            cqm.set_objective((*it.multi_index, x) for x in it if x)

        return cqm

    def mutual_information_cqm(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        *,
        num_features: int,
        alpha: float = 0.5,
        strict: bool = True,
        conditional: bool = True,
        discrete_features: typing.Union[str, npt.ArrayLike] = "auto",
        discrete_target: bool = False,
        copy: bool = True,
        n_neighbors: int = 4,
        n_jobs: typing.Union[None, int] = None,
        random_state: typing.Union[None, int] = None,
    ) -> dimod.ConstrainedQuadraticModel:
        """Build a constrained quadratic model for feature selection.

        If ``conditional`` is True then the conditional mutual information
        criterion from [2] is used, and if ``conditional`` is False then
        mutual information based criterion from [1] is used.

        For computation of mutual information and conditional mutual information


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
            conditional: bool, default=True
                Whether to compute the off-diagonal terms using the conditional mutual
                information or joint mutual information
            discrete_features:
                See :func:`sklearn.feature_selection.mutual_info_regression`
            discrete_target: bool, default=False
                Whether the target variable `y` is discrete
            n_neighbors:
                See :func:`sklearn.feature_selection.mutual_info_regression`
            copy:
                See :func:`sklearn.feature_selection.mutual_info_regression`
            random_state:
                See :func:`sklearn.feature_selection.mutual_info_regression`
            n_jobs: int, default=None
                The number of parallel jobs to run for the conditional mutual information
                computation. The parallelization is over the columns of `X`.
                ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
                ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
                for more details.

        Returns:
            A constrained quadratic model.

        References:
        .. [1] Peng, F. Long, and C. Ding. Feature selection based on mutual information criteria
               of max-dependency, max-relevance, and min-redundancy. IEEE Transactions on pattern
               analysis and machine intelligence, 27(8):1226–1238, 2005.
        .. [2] X. V. Nguyen, J. Chan, S. Romano, and J. Bailey. Effective global approaches for
               mutual information based feature selection. In Proceedings of the 20th ACM SIGKDD
               international conference on Knowledge discovery and data mining,
               pages 512–521. ACM, 2014.
        """

        self._check_params(X, y, alpha, num_features)

        cqm = dimod.ConstrainedQuadraticModel()
        cqm.add_variables(dimod.BINARY, X.shape[1])

        # add the k-hot constraint
        cqm.add_constraint(
            ((v, 1) for v in cqm.variables),
            '==' if strict else '<=',
            num_features,
            label=f"{num_features}-hot",
        )

        mi = estimate_mi_matrix(
            X, y,
            discrete_features=discrete_features,
            discrete_target=discrete_target,
            n_neighbors=n_neighbors, copy=copy,
            random_state=random_state, n_jobs=n_jobs,
            conditional=conditional)

        if conditional:
            # method from [2]_.
            np.multiply(mi, -1, out=mi)
        else:
            # method from [1]_.
            # mutpliypling off-diagonal ones with -1
            diagonal = -np.diag(mi).copy()
            # mutpliypling off-diagonal ones with alpha
            np.multiply(mi, alpha, out=mi)
            np.fill_diagonal(mi, diagonal)

        it = np.nditer(mi, flags=['multi_index'], op_flags=[['readonly']])
        cqm.set_objective((*it.multi_index, x) for x in it if x)
        return cqm

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        *,
        alpha: typing.Optional[float] = None,
        num_features: typing.Optional[int] = None,
        time_limit: typing.Optional[float] = None,
        **kwargs
    ) -> SelectFromQuadraticModel:
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
            This instance of `SelectFromQuadraticModel`.
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

        # time_limit is checked by the LeapHybridCQMSampelr

        # if we already have fewer features than requested, just return
        if num_features >= X.shape[1]:
            self._mask = np.ones(X.shape[1], dtype=bool)
            return self

        if self.method == "correlation":
            cqm = self.correlation_cqm(X, y, num_features=num_features, alpha=alpha)
        elif self.method == "mutual_information":
            cqm = self.mutual_information_cqm(
                X, y, num_features=num_features, alpha=alpha, **kwargs)
        else:
            raise ValueError(f"only methods {self.acceptable_methods} are implemented")

        try:
            sampler = LeapHybridCQMSampler()
        except (ConfigFileError, SolverAuthenticationError) as e:
            raise RuntimeError(
                f"""Instantiation of a Leap hybrid solver failed with an {e} error.

                See https://docs.ocean.dwavesys.com/en/stable/overview/sapi.html for configuring
                access to Leap’s solvers.
                """
            )

        sampleset = sampler.sample_cqm(cqm, time_limit=self.time_limit,
                                       label=f"{self.__module__}.{type(self).__qualname__}")

        filtered = sampleset.filter(lambda d: d.is_feasible)

        if len(filtered) == 0:
            raise RuntimeError("no feasible solutions found by the hybrid solver")

        lowest = filtered.first.sample

        self._mask = np.fromiter((lowest[v] for v in cqm.variables),
                                 count=cqm.num_variables(), dtype=bool)

        return self

    def unfit(self):
        """Undo a previously executed ``fit`` method."""
        del self._mask

    @staticmethod
    def _check_params(X: npt.ArrayLike,
                      y: npt.ArrayLike,
                      alpha: float,
                      num_features: int):
        X = np.atleast_2d(np.asarray(X))
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2-dimensional array-like")

        if y.ndim != 1:
            raise ValueError("y must be a 1-dimensional array-like")

        if y.shape[0] != X.shape[0]:
            raise ValueError(
                f"requires: X.shape[0] == y.shape[0] but {X.shape[0]} != {y.shape[0]}")

        if not 0 <= alpha <= 1:
            raise ValueError(f"alpha must be between 0 and 1, given {alpha}")

        if num_features <= 0:
            raise ValueError(f"num_features must be a positive integer, given {num_features}")

        if X.shape[0] <= 1:
            raise ValueError("X must have at least two rows")


def estimate_mi_matrix(
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    discrete_features: typing.Union[str, npt.ArrayLike] = "auto",
    discrete_target: bool = False,
    n_neighbors: int = 4,
    conditional: bool = True,
    copy: bool = True,
    random_state: typing.Union[None, int] = None,
    n_jobs: typing.Union[None, int] = None,
) -> npt.ArrayLike:
    """
    For the feature array `X` and the target array `y` computes
    the matrix of (conditional) mutual information interactions.
    The matrix is defined as follows:

    `M_(i, i) = I(x_i; y)`

    If `conditional = True`, then the off-diagonal terms are computed:

    `M_(i, j) = (I(x_i; y| x_j) + I(x_j; y| x_i)) / 2`

    Otherwise

    `M_(i, j) = I(x_i; x_j)`

    Computation of I(x; y) uses modified scikit-learn methods.
    The computation of I(x; y| z) is based on
    https://github.com/jannisteunissen/mutual_information and [3]_.

    The method can be computationally expensive for a large number of features (> 1000) and
    a large number of samples (> 100000). In this case, it can be advisable to downsample the
    dataset.

    The estimation methods are linear in the number of outcomes (labels) of discrete distributions.
    It may be beneficial to treat the discrete distrubitions with a large number of labels as
    continuous distributions.

    Args:
        X: See :func:`sklearn.feature_selection.mutual_info_regression`

        y: See :func:`sklearn.feature_selection.mutual_info_regression`

        conditional: bool, default=True
            Whether to compute the off-diagonal terms using the conditional mutual
            information or joint mutual information

        discrete_features: See :func:`sklearn.feature_selection.mutual_info_regression`

        discrete_target: bool, default=False
            Whether the target variable `y` is discrete

        n_neighbors: See :func:`sklearn.feature_selection.mutual_info_regression`

        copy: See :func:`sklearn.feature_selection.mutual_info_regression`

        random_state: See :func:`sklearn.feature_selection.mutual_info_regression`

        n_jobs: int, default=None
            The number of parallel jobs to run for the conditional mutual information
            computation. The parallelization is over the columns of `X`.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

    Returns:
        mi_matrix : ndarray, shape (n_features, n_features)
        Interaction matrix between the features using (conditional) mutual information.
        A negative value will be replaced by 0.

    References:
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    .. [2] B. C. Ross "Mutual Information between Discrete and Continuous
           Data Sets". PLoS ONE 9(2), 2014.
    .. [3] Mesner, Octavio César, and Cosma Rohilla Shalizi. "Conditional
           mutual information estimation for mixed, discrete and continuous
           data." IEEE Transactions on Information Theory 67.1 (2020): 464-484.
    """
    X, y = check_X_y(X, y, accept_sparse="csc", y_numeric=not discrete_target)
    n_samples, n_features = X.shape

    if isinstance(discrete_features, (str, bool)):
        if isinstance(discrete_features, str):
            if discrete_features == "auto":
                discrete_features = issparse(X)
            else:
                raise ValueError("Invalid string value for discrete_features.")
        discrete_mask = np.empty(n_features, dtype=bool)
        discrete_mask.fill(discrete_features)
    else:
        discrete_features = check_array(discrete_features, ensure_2d=False)
        if np.issubdtype(discrete_features.dtype, bool):
            discrete_mask = np.zeros(n_features, dtype=bool)
            discrete_mask[discrete_features] = True
        else:
            discrete_mask = discrete_features

    continuous_mask = ~discrete_mask
    if np.any(continuous_mask) and issparse(X):
        raise ValueError("Sparse matrix `X` can't have continuous features.")

    rng = check_random_state(random_state)
    if np.any(continuous_mask):
        X = X.astype(np.float64, copy=copy)
        X[:, continuous_mask] = scale(
            X[:, continuous_mask], with_mean=False, copy=False
        )

        # Add small noise to continuous features as advised in Kraskov et. al.
        means = np.maximum(1, np.mean(np.abs(X[:, continuous_mask]), axis=0))
        X[:, continuous_mask] += (
            1e-10
            * means
            * rng.standard_normal(size=(n_samples, np.sum(continuous_mask)))
        )

    if not discrete_target:
        y = scale(y, with_mean=False)
        y += (
            1e-10
            * np.maximum(1, np.mean(np.abs(y)))
            * rng.standard_normal(size=n_samples)
        )

    mi_matrix = np.zeros((n_features, n_features), dtype=np.float64)
    # Computing the diagonal terms
    diagonal_vals = Parallel(n_jobs=n_jobs)(
        delayed(_compute_mi)(x, y, discrete_feature, discrete_target, n_neighbors)
        for x, discrete_feature in zip(_iterate_columns(X), discrete_mask)
    )
    np.fill_diagonal(mi_matrix, diagonal_vals)
    # Computing the off-diagonal terms
    off_diagonal_iter = combinations(zip(_iterate_columns(X), discrete_mask), 2)
    if conditional:
        off_diagonal_vals = Parallel(n_jobs=n_jobs)(
            delayed(_compute_cmi_distance)
            (xi, xj, y, discrete_feature_i, discrete_feature_j, discrete_target, n_neighbors)
            for (xi, discrete_feature_i), (xj, discrete_feature_j) in off_diagonal_iter)
    else:
        off_diagonal_vals = Parallel(n_jobs=n_jobs)(
            delayed(_compute_mi)
            (xi, xj, discrete_feature_i, discrete_feature_j, n_neighbors)
            for (xi, discrete_feature_i), (xj, discrete_feature_j) in off_diagonal_iter)

    off_diagonal_val = iter(off_diagonal_vals)
    for i, j in combinations(range(n_features), 2):
        mi_matrix[i, j] = mi_matrix[j, i] = next(off_diagonal_val)
    return mi_matrix
