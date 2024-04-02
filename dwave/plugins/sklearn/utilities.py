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

from itertools import combinations
import typing

from joblib import Parallel, delayed, cpu_count
import numpy as np
import numpy.typing as npt
from scipy.sparse import issparse
from scipy.special import digamma
from scipy.stats import entropy
from sklearn.feature_selection._mutual_info import _compute_mi, _iterate_columns
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_X_y

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

    
def estimate_mi_matrix(    
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    discrete_features: typing.Union[str, npt.ArrayLike]="auto",
    discrete_target: bool = False,
    n_neighbors: int = 4,
    conditional: bool = True,
    copy: bool = True,
    random_state: typing.Union[None, int] = None,
    n_workers: int = 1,
    ) -> npt.ArrayLike:
    """
    For the feature array `X` and the target array `y` computes
    the matrix of (conditional) mutual information interactions. 
    
    cmi_{i, j} = I(x_i; y)
    
    If `conditional = True`, then the off-diagonal terms are computed:
    
    cmi_{i, j} = (I(x_i; y| x_j) + I(x_j; y| x_i)) / 2
    
    Otherwise
    
    cmi_{i, j} = I(x_i; x_j)
    
    Computation of I(x; y) uses the scikit-learn implementation, i.e.,
    :func:`sklearn.feature_selection._mutual_info._estimate_mi`. The computation 
    of I(x; y| z) is based on
    https://github.com/jannisteunissen/mutual_information

    Args:
        X: See :func:`sklearn.feature_selection._mutual_info._estimate_mi`
        
        y: See :func:`sklearn.feature_selection._mutual_info._estimate_mi`
        
        conditional: bool, default=True
            Whether to compute the off-diagonal terms using the conditional mutual
            information or joint mutual information

        discrete_features: See :func:`sklearn.feature_selection._mutual_info._estimate_mi`
        
        discrete_target: See :func:`sklearn.feature_selection._mutual_info._estimate_mi`
        
        n_neighbors: See :func:`sklearn.feature_selection._mutual_info._estimate_mi`
        
        copy: See :func:`sklearn.feature_selection._mutual_info._estimate_mi`
        
        random_state: See :func:`sklearn.feature_selection._mutual_info._estimate_mi`

        n_workers: int, default=1
            Number of workers for parallel computation on the cpu       
        
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
        if discrete_features.dtype != "bool":
            discrete_mask = np.zeros(n_features, dtype=bool)
            discrete_mask[discrete_features] = True
        else:
            discrete_mask = discrete_features

    continuous_mask = ~discrete_mask
    if np.any(continuous_mask) and issparse(X):
        raise ValueError("Sparse matrix `X` can't have continuous features.")

    rng = check_random_state(random_state)
    if np.any(continuous_mask):
        if copy:
            X = X.copy()

        X[:, continuous_mask] = scale(
            X[:, continuous_mask], with_mean=False, copy=False
        )

        # Add small noise to continuous features as advised in Kraskov et. al.
        X = X.astype(np.float64, copy=False)
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
    
    n_features = X.shape[1]
    max_n_workers = cpu_count()-1
    if n_workers > max_n_workers:
        n_workers = max_n_workers
        Warning(f"Specified number of workers {n_workers} is larger than the number of cpus."
                f"Will use only {max_n_workers}.")
    
    mi_matrix = np.zeros((n_features, n_features), dtype=np.float64)    

    off_diagonal_vals = Parallel(n_jobs=n_workers)(
        delayed(_compute_off_diagonal_mi)
        (xi, xj, y, discrete_feature_i, discrete_feature_j, discrete_target, n_neighbors, conditional)
        for (xi, discrete_feature_i), (xj, discrete_feature_j) in combinations(zip(_iterate_columns(X), discrete_mask), 2)
        )
    # keeping the discrete masks since the functions support it
    diagonal_vals = Parallel(n_jobs=n_workers)(
        delayed(_compute_mi)
        (xi, y, discrete_feature_i, discrete_target, n_neighbors)
        for xi, discrete_feature_i in zip(_iterate_columns(X), discrete_mask)
        )
    np.fill_diagonal(mi_matrix, diagonal_vals)
    off_diagonal_val = iter(off_diagonal_vals)
    if conditional:
        # Although we need to compute `(I(X_j; Y| X_i) + I(X_i; Y| X_j))/2`
        # we can avoid the computation of the second term using the formula:
        # `I(X_j; Y| X_i) = I(X_i; Y| X_j) + I(X_j; Y) - I(X_i; Y)`
        for (i, d_i), (j, d_j) in combinations(enumerate(diagonal_vals), 2):
            val = next(off_diagonal_val)            
            val += max(val + (d_j - d_i), 0)
            mi_matrix[i, j] = mi_matrix[j, i] = val/2
    else:
        for i, j in combinations(range(n_features), 2):
            mi_matrix[i, j] = mi_matrix[j, i] = next(off_diagonal_val)
    return mi_matrix


def _compute_off_diagonal_mi(
    xi: npt.ArrayLike,
    xj: npt.ArrayLike,
    y: npt.ArrayLike,
    discrete_feature_i: bool,
    discrete_feature_j: bool,
    discrete_target: bool,
    n_neighbors: int = 4,
    conditional: bool = True,
    ):
    """
    Computing a distance `d` between features `x_i` and `x_j`.
    If `conditional = True`, then conditional mutual infomation is used
    `I(x_i; y | x_j)`.
    
    If `conditonal = False` then mutual information is used 
    `I(x_i; x_j)`.    
    
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
    if conditional:
        if discrete_feature_i and discrete_feature_j and discrete_target:
            return _compute_cmi_d(xi, xj, y)
        else:
            # TODO: consider adding treatement of mixed discrete
            # and continuous variables
            return _compute_cmi_c(xi, xj, y, n_neighbors)
    else:
        return _compute_mi(xi, xj, discrete_feature_i, discrete_feature_j, n_neighbors)


def _compute_cmi_d(
    xi: npt.ArrayLike,
    xj: npt.ArrayLike,
    y: npt.ArrayLike):
    """
    Computes conditional mutual infomation `I(x_i; y | x_j)`.
    All random variables `x_i`, `x_j` and `y` are discrete.
    
    Adpated from https://github.com/dwave-examples/mutual-information-feature-selection
    """
    
    # Computing joint probability distribution for the features and the target
    unique_labels = [np.hstack((np.unique(col), np.inf)) for col in (xi, xj, y)]
    prob, _ = np.histogramdd(np.vstack((xi, xj, y)).T, bins=unique_labels)
    
    cmi_ij = (
        entropy(prob.sum(axis=2).flatten()) # H(x_i, x_j)
        +entropy(prob.sum(axis=0).flatten()) # H(x_j, y)
        -entropy(prob.sum(axis=(0,2))) # H(x_j)
        -entropy(prob.flatten()) # H(x_i, x_j, y)
    )
    return cmi_ij

    
def _compute_cmi_c(
    xi: npt.ArrayLike,
    xj: npt.ArrayLike,
    y: npt.ArrayLike,
    n_neighbors: int = 4):
    """
    Computes conditional mutual infomation `(I(x_i; y | x_j)`.
    At least one random variables from `x_i`, `x_j` and `y` is continuous.
    
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

    cmi_ij = (
        digamma(n_neighbors)
        +np.mean(digamma(n_j))
        -np.mean(digamma(n_ij))        
        -np.mean(digamma(n_jy)))
    return max(0, cmi_ij)


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
