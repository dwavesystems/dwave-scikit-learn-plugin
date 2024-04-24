# The following traversal code is adapted from two sources
# I. Scikit-learn implementation for mutual information computation
# II. Computation of conditional mutual information in the repo
#       https://github.com/jannisteunissen/mutual_information
#
# I. Methods and algorithms from `sklearn.feature_selection._mutual_info.py`
#
# Author: Nikolay Mayorov <n59_ru@hotmail.com>
# License: 3-clause BSD
#
# II. Modifications in https://github.com/jannisteunissen/mutual_information
# BSD 3-Clause License
#
# Copyright (c) 2022, Jannis Teunissen
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Modifications are distributed under the Apache 2.0 Software license.
#
# Copyright 2023 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import combinations
import typing

import numpy as np
import numpy.typing as npt

from scipy.sparse import issparse
from scipy.special import digamma
from scipy.stats import entropy
from sklearn.metrics.cluster import mutual_info_score
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_array, check_X_y


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


def _compute_cmi_distance(
    xi: npt.ArrayLike,
    xj: npt.ArrayLike,
    y: npt.ArrayLike,
    discrete_feature_i: bool,
    discrete_feature_j: bool,
    discrete_target: bool,
    n_neighbors: int = 4,
) -> float:
    """
    Computes a distance `d` based on the conditional mutual information
    between features `x_i` and `x_j`: `d = (I(x_i; y | x_j)+I(x_j; y | x_i))/2`.
    Args:
        xi:
            A feature vector formatted as a numerical 1D array-like.
        xj:
            A feature vector formatted as a numerical 1D array-like.
        y:
            Target vector formatted as a numerical 1D array-like.
        discrete_feature_i:
            Whether to consider `xi` as a discrete variable.
        discrete_feature_j:
            Whether to consider `xj` as a discrete variable.
        discrete_target:
            Whether to consider `y` as a discrete variable.
        n_neighbors:
            Number of neighbors to use for MI estimation for continuous variables,
            see [1]_ and [2]_. Higher values reduce variance of the estimation, but
            could introduce a bias.
    Returns:
        Distance between features based on conditional mutual information.
    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    .. [2] B. C. Ross "Mutual Information between Discrete and Continuous
           Data Sets". PLoS ONE 9(2), 2014.
    .. [3] Mesner, Octavio César, and Cosma Rohilla Shalizi. "Conditional
           mutual information estimation for mixed, discrete and continuous
           data." IEEE Transactions on Information Theory 67.1 (2020): 464-484.
    """
    if discrete_feature_i and discrete_feature_j and discrete_target:
        ans = _compute_cmip_d(xi, xj, y)
    elif discrete_target:
        ans = _compute_cmip_ccd(xi, xj, y, n_neighbors)
    elif discrete_feature_i and (not discrete_target):
        ans = _compute_cmip_cdc(xj, xi, y, n_neighbors)
    elif (not discrete_feature_i) and discrete_feature_j and (not discrete_target):
        ans = _compute_cmip_cdc(xi, xj, y, n_neighbors)
    else:
        ans = _compute_cmip_c(xi, xj, y, n_neighbors)
    return np.mean(ans)


def _compute_cmip_d(
        xi: npt.ArrayLike,
        xj: npt.ArrayLike,
        y: npt.ArrayLike) -> typing.Tuple[float]:
    """
    Computes conditional mutual information pair `I(x_i; y | x_j)` and `I(x_j; y | x_i)`.
    All random variables `x_i`, `x_j` and `y` are discrete.

    Adapted from https://github.com/dwave-examples/mutual-information-feature-selection
    Args:
        xi:
            A feature vector formatted as a numerical 1D array-like.
        xj:
            A feature vector formatted as a numerical 1D array-like.
        y:
            Target vector formatted as a numerical 1D array-like.
        n_neighbors:
            Number of neighbors to use for MI estimation for continuous variables,
            see [1]_ and [2]_. Higher values reduce variance of the estimation, but
            could introduce a bias.
    Returns:
        Pair of conditional mutual information values between two features and the target vector.
    """
    # Computing joint probability distribution for the features and the target
    bin_boundaries = [np.hstack((np.unique(col), np.inf)) for col in (xi, xj, y)]
    prob, _ = np.histogramdd(np.vstack((xi, xj, y)).T, bins=bin_boundaries)

    # Computing entropy
    Hijy = entropy(prob.flatten())  # H(x_i, x_j, y)
    Hij = entropy(prob.sum(axis=2).flatten())  # H(x_i, x_j)
    cmi_ij = (
        Hij-Hijy
        + entropy(prob.sum(axis=0).flatten())  # H(x_j, y)
        - entropy(prob.sum(axis=(0, 2)))  # H(x_j)
    )
    cmi_ji = (
        Hij-Hijy
        + entropy(prob.sum(axis=1).flatten())  # H(x_i, y)
        - entropy(prob.sum(axis=(1, 2)))  # H(x_i)
    )
    return cmi_ij, cmi_ji


def _compute_cmip_c(
        xi: npt.ArrayLike,
        xj: npt.ArrayLike,
        y: npt.ArrayLike,
        n_neighbors: int = 4) -> typing.Tuple[float]:
    """
    Computes conditional mutual information pair `I(x_i; y | x_j)`
    and `I(x_j; y | x_i)`. All random variables `x_i`, `x_j` and `y`
    are assumed to be continuous.

    Adapted from https://github.com/jannisteunissen/mutual_information
    :func:`sklearn.feature_selection.mutual_info_regression` and
    :func:`sklearn.feature_selection.mutual_info_classif`.

    Args:
        xi:
            A feature vector formatted as a numerical 1D array-like.
        xj:
            A feature vector formatted as a numerical 1D array-like.
        y:
            Target vector formatted as a numerical 1D array-like.
        n_neighbors:
            Number of neighbors to use for MI estimation for continuous variables,
            see [1]_ and [2]_. Higher values reduce variance of the estimation, but
            could introduce a bias.
    Returns:
        Pair of conditional mutual information values between two features and the target vector.
    """
    xi = xi.reshape((-1, 1))
    xj = xj.reshape((-1, 1))
    y = y.reshape((-1, 1))

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    radius = _get_radius_k_neighbours(np.hstack((xi, xj, y)),
                                      n_neighbors=n_neighbors)

    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    n_j = _num_points_within_radius(xj, radius)
    n_ij = _num_points_within_radius(np.hstack((xi, xj)), radius)
    n_jy = _num_points_within_radius(np.hstack((y, xj)), radius)

    n_i = _num_points_within_radius(xi, radius)
    n_iy = _num_points_within_radius(np.hstack((y, xi)), radius)

    return _get_cmi_pair_from_estimates([n_neighbors], n_ij, n_j, n_jy, n_i, n_iy)


def _compute_cmip_ccd(
        xi: npt.ArrayLike,
        xj: npt.ArrayLike,
        y: npt.ArrayLike,
        n_neighbors: int = 4) -> typing.Tuple[float]:
    """
    Computes conditional mutual information pair `I(x_i; y | x_j)`
    and `I(x_j; y | x_i)`. Random variables `x_i`, `x_j` are assumed
    to be continuous, while `y` is assumed to be discrete.

    Adapted from https://github.com/jannisteunissen/mutual_information
    :func:`sklearn.feature_selection.mutual_info_regression` and
    :func:`sklearn.feature_selection.mutual_info_classif`.
    Args:
        xi:
            A feature vector formatted as a numerical 1D array-like.
        xj:
            A feature vector formatted as a numerical 1D array-like.
        y:
            Target vector formatted as a numerical 1D array-like.
        n_neighbors:
            Number of neighbors to use for MI estimation for continuous variables,
            see [1]_ and [2]_. Higher values reduce variance of the estimation, but
            could introduce a bias.
    Returns:
        Pair of conditional mutual information values between two features and the target vector.
    """
    xi = xi.reshape((-1, 1))
    xj = xj.reshape((-1, 1))

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    radius, label_counts, k_all = _get_radius_k_neighbours_d(np.hstack((xi, xj)), y,
                                                             n_neighbors=n_neighbors)

    # Ignore points with unique labels.
    mask = label_counts > 1
    # label_counts = label_counts[mask]
    k_all = k_all[mask]
    xi = xi[mask]
    xj = xj[mask]
    y = y[mask]
    radius = radius[mask]
    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    # continuous
    n_i = _num_points_within_radius(xi, radius)
    n_j = _num_points_within_radius(xj, radius)
    n_ij = _num_points_within_radius(np.hstack((xi, xj)), radius)
    # mixed discrete/continuous estimates
    n_iy = _num_points_within_radius_cd(xi, y, radius)
    n_jy = _num_points_within_radius_cd(xj, y, radius)

    return _get_cmi_pair_from_estimates(k_all, n_ij, n_j, n_jy, n_i, n_iy)


def _compute_cmip_cdc(
        xi: npt.ArrayLike,
        xj: npt.ArrayLike,
        y: npt.ArrayLike,
        n_neighbors: int = 4) -> typing.Tuple[float]:
    """
    Computes conditional mutual information pair `I(x_i; y | x_j)`
    and `I(x_j; y | x_i)`. Random variables `x_i`, `y` are assumed
    to be continuous, while `x_j` is assumed to be discrete.

    Adapted from https://github.com/jannisteunissen/mutual_information
    :func:`sklearn.feature_selection.mutual_info_regression` and
    :func:`sklearn.feature_selection.mutual_info_classif`.
    Args:
        xi:
            A feature vector formatted as a numerical 1D array-like.
        xj:
            A feature vector formatted as a numerical 1D array-like.
        y:
            Target vector formatted as a numerical 1D array-like.
        n_neighbors:
            Number of neighbors to use for MI estimation for continuous variables,
            see [1]_ and [2]_. Higher values reduce variance of the estimation, but
            could introduce a bias.
    Returns:
        Pair of conditional mutual information values between two features and the target vector.
    """
    xi = xi.reshape((-1, 1))
    y = y.reshape((-1, 1))

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    radius, label_counts, k_all = _get_radius_k_neighbours_d(np.hstack((xi, y)), xj,
                                                             n_neighbors=n_neighbors)

    # Ignore points with unique labels.
    mask = label_counts > 1
    label_counts = label_counts[mask]
    k_all = k_all[mask]
    xi = xi[mask]
    xj = xj[mask]
    y = y[mask]
    radius = radius[mask]

    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    # continuous
    n_i = _num_points_within_radius(xi, radius)
    n_iy = _num_points_within_radius(np.hstack((xi, y)), radius)
    # mixed coninuous/discrete estimates
    n_ij = _num_points_within_radius_cd(xi, xj, radius)
    n_jy = _num_points_within_radius_cd(y, xj, radius)
    # discrete estimates
    n_j = label_counts

    return _get_cmi_pair_from_estimates(k_all, n_ij, n_j, n_jy, n_i, n_iy)


def _get_cmi_pair_from_estimates(
        k_all: npt.ArrayLike, n_ij: npt.ArrayLike, n_j: npt.ArrayLike,
        n_jy: npt.ArrayLike, n_i: npt.ArrayLike, n_iy: npt.ArrayLike) -> typing.Tuple[float]:
    """Get an estimate from nearest neighbors counts"""
    common_terms = np.mean(digamma(k_all))-np.mean(digamma(n_ij))
    cmi_ij = common_terms+np.mean(digamma(n_j))-np.mean(digamma(n_jy))
    cmi_ji = common_terms+np.mean(digamma(n_i))-np.mean(digamma(n_iy))
    return max(0, cmi_ij), max(0, cmi_ji)


def _compute_mi(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    discrete_feature_x: bool,
    discrete_feature_y: bool,
    n_neighbors: int = 4,
):
    """
    Compute mutual information between features `I(x_i; x_j)`.

    Adapted from :func:`sklearn.feature_selection.mutual_info_regression` and
    :func:`sklearn.feature_selection.mutual_info_classif`.

    Args:
        x, y:
            Feature vectors each formatted as a numerical 1D array-like.
        discrete_feature_x, discrete_feature_y:
            Whether to consider `x` and `y` as discrete variables.
        n_neighbors:
            Number of neighbors to use for MI estimation for continuous variables,
            see [1]_ and [2]_. Higher values reduce variance of the estimation, but
            could introduce a bias.
    Returns:
        Mutual information between feature vectors x and y.
    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    .. [2] B. C. Ross "Mutual Information between Discrete and Continuous
           Data Sets". PLoS ONE 9(2), 2014.
    """
    if discrete_feature_y and discrete_feature_x:
        return mutual_info_score(x, y)
    elif discrete_feature_x:
        return _compute_mi_cd(y, x, n_neighbors=n_neighbors)
    elif discrete_feature_y:
        return _compute_mi_cd(x, y, n_neighbors=n_neighbors)
    else:
        return _compute_mi_cc(x, y, n_neighbors=n_neighbors)


def _compute_mi_cc(x: npt.ArrayLike, y: npt.ArrayLike, n_neighbors: int) -> float:
    """Computes mutual information `I(x; y)`. Random variables `x`, `y` are assumed
    to be continuous.

    Adapted from :func:`sklearn.feature_selection.mutual_info_regression` and
    :func:`sklearn.feature_selection.mutual_info_classif`.
    Args:
        x, y:
            Feature vectors each formatted as a numerical 1D array-like.
        n_neighbors:
            Number of neighbors to use for MI estimation for continuous variables,
            see [1]_ and [2]_. Higher values reduce variance of the estimation, but
            could introduce a bias.
    Returns:
        Mutual information between feature vectors x and y.
    """
    n_samples = x.size

    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))

    # Here we rely on NearestNeighbors to select the fastest algorithm.
    radius = _get_radius_k_neighbours(np.hstack((x, y)), n_neighbors)

    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    nx = _num_points_within_radius(x, radius)
    ny = _num_points_within_radius(y, radius)

    mi = (
        digamma(n_samples)
        + digamma(n_neighbors)
        - np.mean(digamma(nx))
        - np.mean(digamma(ny))
    )

    return max(0, mi)


def _compute_mi_cd(x: npt.ArrayLike, y: npt.ArrayLike, n_neighbors: int) -> float:
    """Computes mutual information `I(x; y)`. Random variable `x`, is assumed
    to be continuous, while `y` is assumed to be discrete.

    Adapted from :func:`sklearn.feature_selection.mutual_info_regression` and
    :func:`sklearn.feature_selection.mutual_info_classif`.
    Args:
        x, y:
            Feature vectors each formatted as a numerical 1D array-like.
        n_neighbors:
            Number of neighbors to use for MI estimation for continuous variables,
            see [1]_ and [2]_. Higher values reduce variance of the estimation, but
            could introduce a bias.
    Returns:
        Mutual information between feature vectors x and y.
    """
    radius, label_counts, k_all = _get_radius_k_neighbours_d(x, y, n_neighbors=n_neighbors)

    # Ignore points with unique labels.
    mask = label_counts > 1
    n_samples = np.sum(mask)
    label_counts = label_counts[mask]
    k_all = k_all[mask]
    x = x[mask]
    radius = radius[mask]

    m_all = _num_points_within_radius(x.reshape(-1, 1), radius)

    mi = (
        digamma(n_samples)
        + np.mean(digamma(k_all))
        - np.mean(digamma(label_counts))
        - np.mean(digamma(m_all))
    )

    return max(0, mi)


def _get_radius_k_neighbours(X: npt.ArrayLike, n_neighbors: int = 4) -> npt.ArrayLike:
    """
    Determine the smallest radius around `X` containing `n_neighbors` neighbors
    Inspired by :func:`sklearn.feature_selection.mutual_info_regression` and
    :func:`sklearn.feature_selection.mutual_info_classif`.

    Args:
        X:
            Array of features.
        n_neighbors:
            Number of nearest neighbors.
    Returns:
        Vector of radii defined by the k nearest neighbors.
    """
    nn = NearestNeighbors(metric="chebyshev", n_neighbors=n_neighbors)
    return _get_radius_from_nn(X, nn)


def _get_radius_k_neighbours_d(
    c: npt.ArrayLike,
    d: npt.ArrayLike,
    n_neighbors: int = 4
) -> typing.Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """
    Determine smallest radius around `c` and `d` containing `n_neighbors`.

    Inspired by :func:`sklearn.feature_selection.mutual_info_regression` and
    :func:`sklearn.feature_selection.mutual_info_classif`.
    Args:
        c:
            Array of feature vectors.
        d:
            Vector of discrete features formatted as a numerical 1D array-like.
        n_neighbors:
            The number of nearest neighbors.
    Returns:
        radius:
            Vector of radii defined by the k nearest neighbors.
        label_counts:
            Label counts of the discrete feature vector.
        k_all:
            Array of the number of points within a radius.
    """
    d = d.flatten()
    n_samples = d.shape[0]
    c = c.reshape((n_samples, -1))

    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)
    nn = NearestNeighbors(metric="chebyshev")
    for label in np.unique(d):
        mask = d == label
        count = np.sum(mask)
        if count > 1:
            k = min(n_neighbors, count - 1)
            nn.set_params(n_neighbors=k)
            radius[mask] = _get_radius_from_nn(c[mask], nn)
            k_all[mask] = k
        label_counts[mask] = np.sum(mask)
    return radius, label_counts, k_all


def _get_radius_from_nn(X: npt.ArrayLike, nn: NearestNeighbors) -> npt.ArrayLike:
    """
    Get the radius of nearest neighbors in `nn` model with dataset `X`.

    Inspired by :func:`sklearn.feature_selection.mutual_info_regression` and
    :func:`sklearn.feature_selection.mutual_info_classif`.
    Args:
        X:
            Array of features.
        nn:
            Instance of the nearest neighbors class.
    Returns:
        Vector of radii defined by the k nearest neighbors.
    """
    nn.fit(X)
    r = nn.kneighbors()[0]
    return np.nextafter(r[:, -1], 0)


def _num_points_within_radius(x: npt.ArrayLike, radius: npt.ArrayLike) -> npt.ArrayLike:
    """For each point, determine the number of other points within a given radius
    Function from https://github.com/jannisteunissen/mutual_information
    Args:
        X:
            (Continuous) feature array.
        radius:
            Vector of radii.
    Returns:
        Vector containing the number of points within a radius.
    """
    kd = KDTree(x, metric="chebyshev")
    nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
    return np.array(nx)


def _num_points_within_radius_cd(
        c: npt.ArrayLike,
        d: npt.ArrayLike,
        radius: npt.ArrayLike) -> npt.ArrayLike:
    """
    For each point, determine the number of other points within a given radius.

    Inspired by :func:`sklearn.feature_selection.mutual_info_regression` and
    :func:`sklearn.feature_selection.mutual_info_classif`.
    Args:
        c:
            (Continuous) feature array.
        d:
            Discrete feature vector formatted as a numerical 1D array-like
        radius:
            Vector of radii.
    Returns:
        Vector containing the number of points within a radius.
    """
    c = c.reshape((-1, 1))
    n_samples = c.shape[0]
    m_all = np.empty(n_samples)
    for label in np.unique(d):
        mask = d == label
        m_all[mask] = _num_points_within_radius(c[mask], radius[mask])
    return m_all


def _iterate_columns(X, columns=None):
    """Iterate over columns of a matrix.
    Copied from :func:`sklearn.feature_selection._mutual_info`.

    Parameters
    ----------
    X : ndarray or csc_matrix, shape (n_samples, n_features)
        Matrix over which to iterate.

    columns : iterable or None, default=None
        Indices of columns to iterate over. If None, iterate over all columns.

    Yields
    ------
    x : ndarray, shape (n_samples,)
        Columns of `X` in dense format.
    """
    if columns is None:
        columns = range(X.shape[1])

    if issparse(X):
        for i in columns:
            x = np.zeros(X.shape[0])
            start_ptr, end_ptr = X.indptr[i], X.indptr[i + 1]
            x[X.indices[start_ptr:end_ptr]] = X.data[start_ptr:end_ptr]
            yield x
    else:
        for i in columns:
            yield X[:, i]
