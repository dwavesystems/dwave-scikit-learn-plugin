# The following traversal code is adapted from NumPy's implementation.

# Copyright (c) 2005-2022, NumPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.
#
#     * Neither the name of the NumPy Developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Modifications are licensed under the Apache 2.0 Software license.

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

import typing

import numpy as np
import numpy.typing as npt
from scipy.sparse import issparse
from scipy.special import digamma
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

__all__ = ["corrcoef", "cov", "dot_2d"]


def corrcoef(x: npt.ArrayLike, *,
             out: typing.Optional[np.ndarray] = None,
             rowvar: bool = True,
             copy: bool = True,
             ) -> np.ndarray:
    """A drop-in replacement for :func:`numpy.corrcoef`.

    This method is modified to avoid unnecessary memory usage when working with
    :class:`numpy.memmap` arrays.
    It does not support the full range of arguments accepted by
    :func:`numpy.corrcoef`.

    Additionally, in the case that a row of ``x`` is fixed, this method
    will return a correlation value of 0 rather than :class:`numpy.nan`.

    Args:
        x: See :func:`numpy.corrcoef`.

        out: Output argument. This must be the exact kind that would be returned
            if it was not used.

        rowvar: See :func:`numpy.corrcoef`.

        copy: If ``True``, ``x`` is not modified by this function.

    Returns:
        See :func:`numpy.corrcoef`.

    """
    c = cov(x, out=out, rowvar=rowvar, copy=copy)
    try:
        d = np.diag(c)
    except ValueError:
        # scalar covariance
        # nan if incorrect value (nan, inf, 0), 1 otherwise
        return c / c
    stddev = np.sqrt(d.real)

    # the places that stddev == 0 are exactly the places that the columns
    # are fixed. We can safely ignore those when dividing
    np.divide(c, stddev[:, None], out=c, where=stddev[:, None] != 0)
    np.divide(c, stddev[None, :], out=c, where=stddev[None, :] != 0)

    # Clip real and imaginary parts to [-1, 1].  This does not guarantee
    # abs(a[i,j]) <= 1 for complex arrays, but is the best we can do without
    # excessive work.
    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):
        np.clip(c.imag, -1, 1, out=c.imag)

    return c


def cov(m: npt.ArrayLike, *,
        out: typing.Optional[np.ndarray] = None,
        rowvar: bool = True,
        copy: bool = True,
        ) -> np.ndarray:
    """A drop-in replacement for :func:`numpy.cov`.

    This method is modified to avoid unnecessary memory usage when working with
    :class:`numpy.memmap` arrays.
    It does not support the full range of arguments accepted by
    :func:`numpy.cov`.

    Args:
        m: See :func:`numpy.cov`.

        out: Output argument. This must be the exact kind that would be returned
            if it was not used.

        rowvar: See :func:`numpy.cov`.

        copy: If ``True``, ``x`` is not modified by this function.

    Returns:
        See :func:`numpy.cov`.

    """
    # we want to modify X, so if copy=True we make a copy and re-call
    if copy:
        if hasattr(m, "flush"):
            # we could do a lot of fiddling here, but it's easier to just
            # disallow this case and rely on the user making a modifiable
            # X
            raise ValueError("memmap arrays cannot be copied easily")

        return cov(np.array(m), rowvar=rowvar, copy=False, out=out)

    # handle array-like
    if isinstance(m, np.memmap):
        X = m
    else:
        X = np.atleast_2d(np.asarray(m, dtype=np.result_type(m, np.float64)))

    if X.ndim != 2:
        raise ValueError("X must have 2 dimensions")

    if not rowvar and X.shape[0] != 1:
        X = X.T

    # Get the product of frequencies and weights
    avg = np.average(X, axis=1)

    # Determine the normalization
    fact = max(X.shape[1] - 1, 0)

    X -= avg[:, None]

    if hasattr(m, "flush"):
        X.flush()

    X_T = X.T

    out = dot_2d(X, X_T.conj(), out=out)
    out *= np.true_divide(1, fact)

    if hasattr(out, "flush"):
        out.flush()

    return out


def dot_2d(a: npt.ArrayLike, b: npt.ArrayLike, *,
           out: typing.Optional[np.ndarray] = None,
           chunksize: int = int(1e+9),
           ) -> np.ndarray:
    """A drop-in replacment for :func:`numpy.dot` for 2d arrays.

    This method is modified to avoid unnecessary memory usage when working with
    :class:`numpy.memmap` arrays.

    Args:
        a: See :func:`numpy.dot`. ``a.ndim`` must be 2.
        b: See :func:`numpy.dot`. ``b.ndim`` must be 2.
        out: See :func:`numpy.dot`.
        chunksize: The number of bytes that should be created by each step
            of the multiplication. This is used to keep the total memory
            usage low when multiplying :class:`numpy.memmap` arrays.

    Returns:
        See :func:`numpy.dot`.

    """
    if not isinstance(a, np.memmap):
        a = np.asarray(a)
    if not isinstance(b, np.memmap):
        b = np.asarray(b)

    if a.ndim != 2:
        raise ValueError("a must be a 2d array")
    if b.ndim != 2:
        raise ValueError("b must be a 2d array")

    if out is None:
        out = np.empty((a.shape[0], b.shape[1]), dtype=np.result_type(a, b))
    elif out.shape[0] != a.shape[0] or out.shape[1] != b.shape[1]:
        raise ValueError(f"out must be a ({a.shape[0]}, {b.shape[1]}) array")

    is_memmap = hasattr(out, "flush")

    num_rows = max(chunksize // (out.dtype.itemsize * out.shape[1]), 1)
    for start in range(0, out.shape[0], num_rows):
        np.dot(a[start:start+num_rows, :], b, out=out[start:start+num_rows, :])

        if is_memmap:
            out.flush()

    return out


def _compute_off_diagonal_cmi(
    xi: npt.ArrayLike,
    xj: npt.ArrayLike,
    y: npt.ArrayLike,
    discrete_feature_i: bool,
    discrete_feature_j: bool,
    discrete_target: bool,
    n_neighbors: int = 4,
    ):
    """
    Computing a distance `d` based on the conditional mutual infomation
    between features `x_i` and `x_j`: `d = (I(x_i; y | x_j)+I(x_j; y | x_i))/2`.
    
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
        val = _compute_cmip_d(xi, xj, y)
    elif discrete_target:            
        val = _compute_cmip_ccd(xi, xj, y, n_neighbors)            
    elif discrete_feature_i and (not discrete_target):
        val = _compute_cmip_cdc(xj, xi, y, n_neighbors)
    elif (not discrete_feature_i) and discrete_feature_j and (not discrete_target):
        val = _compute_cmip_cdc(xi, xj, y, n_neighbors)
    else:
        val = _compute_cmip_c(xi, xj, y, n_neighbors)
    return sum(val) / 2


def _compute_off_diagonal_mi(
    xi: npt.ArrayLike,
    xj: npt.ArrayLike,
    discrete_feature_i: bool,
    discrete_feature_j: bool,
    n_neighbors: int = 4,
    ):
    """
    Compute mutual information between features `I(x_i; x_j)`.
    
    References
    ----------
    .. [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
           information". Phys. Rev. E 69, 2004.
    .. [2] B. C. Ross "Mutual Information between Discrete and Continuous
           Data Sets". PLoS ONE 9(2), 2014.
    """
    if discrete_feature_j:
        return mutual_info_classif(
            xi.reshape(-1, 1), xj, discrete_features=[discrete_feature_i], n_neighbors=n_neighbors)
    elif discrete_feature_i:
        return mutual_info_classif(
            xj.reshape(-1, 1), xi, discrete_features=[discrete_feature_j], n_neighbors=n_neighbors)
    else:
        return mutual_info_regression(
            xi.reshape(-1, 1), xj, discrete_features=[discrete_feature_i], n_neighbors=n_neighbors)


def _compute_cmip_d(
    xi: npt.ArrayLike,
    xj: npt.ArrayLike,
    y: npt.ArrayLike) -> typing.Tuple[float]:
    """
    Computes conditional mutual infomation pair `I(x_i; y | x_j)` `I(x_j; y | x_i)`.
    All random variables `x_i`, `x_j` and `y` are discrete.
    
    Adpated from https://github.com/dwave-examples/mutual-information-feature-selection
    """
    
    # Computing joint probability distribution for the features and the target
    bin_boundaries = [np.hstack((np.unique(col), np.inf)) for col in (xi, xj, y)]
    prob, _ = np.histogramdd(np.vstack((xi, xj, y)).T, bins=bin_boundaries)
    Hijy = entropy(prob.flatten())   
    Hij = entropy(prob.sum(axis=2).flatten())
    cmi_ij = (
        Hij # H(x_i, x_j)
        +entropy(prob.sum(axis=0).flatten()) # H(x_j, y)
        -entropy(prob.sum(axis=(0,2))) # H(x_j)
        -Hijy # H(x_i, x_j, y)
    )
    cmi_ji = (
        Hij # H(x_i, x_j)
        +entropy(prob.sum(axis=1).flatten()) # H(x_i, y)
        -entropy(prob.sum(axis=(1,2))) # H(x_i)
        -Hijy # H(x_i, x_j, y)        
    )
    return cmi_ij, cmi_ji


def _compute_cmip_c(
    xi: npt.ArrayLike,
    xj: npt.ArrayLike,
    y: npt.ArrayLike,
    n_neighbors: int = 4) -> typing.Tuple[float]:
    """
    Computes conditional mutual information pair `(I(x_i; y | x_j)`
    and `(I(x_j; y | x_i)`. All random variables `x_i`, `x_j` and `y` 
    are assumed to be continuous.
    
    Adapted from https://github.com/jannisteunissen/mutual_information
    """
    xi = xi.reshape((-1, 1))    
    xj = xj.reshape((-1, 1))
    y = y.reshape((-1, 1))
    
    # Here we rely on NearestNeighbors to select the fastest algorithm.
    nn = NearestNeighbors(metric="chebyshev", n_neighbors=n_neighbors)
    nn.fit(np.hstack((xi, xj, y)))
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)
       
    # KDTree is explicitly fit to allow for the querying of number of
    # neighbors within a specified radius
    n_j = _num_points_within_radius(xj, radius)
    n_ij = _num_points_within_radius(np.hstack((xi, xj)), radius)
    n_jy = _num_points_within_radius(np.hstack((y, xj)), radius)

    n_i = _num_points_within_radius(xi, radius)
    n_iy = _num_points_within_radius(np.hstack((y, xi)), radius)

    cmi_ij = (
        digamma(n_neighbors)
        +np.mean(digamma(n_j))
        -np.mean(digamma(n_ij))        
        -np.mean(digamma(n_jy)))

    cmi_ji = (
        digamma(n_neighbors)
        +np.mean(digamma(n_i))
        -np.mean(digamma(n_ij))        
        -np.mean(digamma(n_iy)))
    return max(0, cmi_ij), max(0, cmi_ji)


def _compute_cmip_ccd(
    xi: npt.ArrayLike,
    xj: npt.ArrayLike,
    y: npt.ArrayLike,
    n_neighbors: int = 4) -> typing.Tuple[float]:
    """
    Computes conditional mutual information pair `(I(x_i; y | x_j)`
    and `(I(x_j; y | x_i)`. Random variables `x_i`, `x_j` are assumed
    to be continuous, while `y` is assumed to be discrete.
    
    Adapted from https://github.com/jannisteunissen/mutual_information and 
    :func:`sklearn.feature_selection._mutual_info._compute_mi_cd`
    """
    xi = xi.reshape((-1, 1))    
    xj = xj.reshape((-1, 1))

    (radius,
     label_counts,
     k_all) = _get_radius_k_neighbours_d(
         xi, xj, y,
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
    n_j = _num_points_within_radius(xj, radius)
    n_ij = _num_points_within_radius(np.hstack((xi, xj)), radius)
    
    # mixed estimates    
    n_iy = _num_points_within_radius_cd(xi, y, radius)
    n_jy = _num_points_within_radius_cd(xj, y, radius)
    
    common_terms = (
        np.mean(digamma(k_all))
        -np.mean(digamma(n_ij)))
    cmi_ij = (
        common_terms
        +np.mean(digamma(n_j))
        -np.mean(digamma(n_jy)))
    cmi_ji = (
        common_terms
        +np.mean(digamma(n_i))
        -np.mean(digamma(n_iy)))
    return max(0, cmi_ij), max(0, cmi_ji)


def _compute_cmip_cdc(
    xi: npt.ArrayLike,
    xj: npt.ArrayLike,
    y: npt.ArrayLike,
    n_neighbors: int = 4) -> typing.Tuple[float]:
    """
    Computes conditional mutual information pair `(I(x_i; y | x_j)`
    and `(I(x_j; y | x_i)`. Random variables `x_i`, `y` are assumed
    to be continuous, while `x_j` is assumed to be discrete.
    
    Adapted from https://github.com/jannisteunissen/mutual_information and
    :func:`sklearn.feature_selection._mutual_info._compute_mi_cd`
    """    
    xi = xi.reshape((-1, 1))
    y = y.reshape((-1, 1))

    (radius,
     label_counts,
     k_all) = _get_radius_k_neighbours_d(
         xi, y, xj,
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
    
    # mixed estimates
    n_ij = _num_points_within_radius_cd(xi, xj, radius)
    n_jy = _num_points_within_radius_cd(y, xj, radius)
    
    # discrete estimates
    n_j = label_counts
    
    common_terms = (
        np.mean(digamma(k_all))
        -np.mean(digamma(n_ij)))
    cmi_ij = (
        common_terms
        +np.mean(digamma(n_j))
        -np.mean(digamma(n_jy)))
    cmi_ji = (
        common_terms
        +np.mean(digamma(n_i))
        -np.mean(digamma(n_iy)))
    return max(0, cmi_ij), max(0, cmi_ji)


def _get_radius_k_neighbours_d(
    c1: npt.ArrayLike,
    c2: npt.ArrayLike,
    d: npt.ArrayLike,
    n_neighbors: int = 4
    ) -> typing.Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
    """
    Determine smallest radius around x containing n_neighbors neighbors
    Inspired by :func:`sklearn.feature_selection._mutual_info._compute_mi_cd`
    """
    c1 = c1.reshape((-1, 1))
    c2 = c2.reshape((-1, 1))
    n_samples = c1.shape[0]
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
            nn.fit(np.hstack((c1[mask], c2[mask])))
            r = nn.kneighbors()[0]
            radius[mask] = np.nextafter(r[:, -1], 0)
            k_all[mask] = k
        label_counts[mask] = np.sum(mask)        
    return radius, label_counts, k_all


def _num_points_within_radius(x: npt.ArrayLike, radius: npt.ArrayLike):
    """For each point, determine the number of other points within a given radius
    Function from https://github.com/jannisteunissen/mutual_information
    
    :param x: ndarray, shape (n_samples, n_dim)
    :param radius: radius, shape (n_samples,)
    :returns: number of points within radius

    """
    kd = KDTree(x, metric="chebyshev")
    nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
    return np.array(nx)


def _num_points_within_radius_cd(
    c: npt.ArrayLike,
    d: npt.ArrayLike,
    radius: npt.ArrayLike) -> npt.ArrayLike:
    """
    For each point, determine the number of other points within a given radius
    Inspired by :func:`sklearn.feature_selection._mutual_info._compute_mi_cd`
    """
    c = c.reshape((-1, 1))
    n_samples = c.shape[0]
    m_all = np.empty(n_samples)
    for label in np.unique(d):
        mask = d == label
        kd = KDTree(c[mask], metric="chebyshev")
        m_all[mask] = kd.query_radius(
            c[mask], radius[mask],
            count_only=True, return_distance=False)
    return m_all


def _iterate_columns(X, columns=None):
    """Iterate over columns of a matrix.
    Copied from :func:`sklearn.feature_selection._mutual_info._iterate_columns`

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
            