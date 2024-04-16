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

# Some tests are adapted from NumPy under the following license.

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

import tempfile
import unittest

import numpy as np

from dwave.plugins.sklearn.utilities import (
    corrcoef, cov, dot_2d,
    _compute_mi,
    _compute_cmip_c, _compute_cmip_d,
    _compute_cmip_cdc, _compute_cmip_ccd
)
from scipy.special import digamma
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.neighbors import KDTree
from sklearn.preprocessing import scale


class TestCorrCoef(unittest.TestCase):
    def test_agreement(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(size=(100, 100))
        for rowvar in (True, False):
            with self.subTest(rowvar=rowvar):
                np.testing.assert_array_equal(
                    corrcoef(X, rowvar=rowvar), np.corrcoef(X, rowvar=rowvar))

    def test_memmap(self):
        # Smoketest for memmap.
        # here isn't really a nice way to test memory usage but it's useful to
        # have this test present for manual testing
        rng = np.random.default_rng(42)

        size = (1_000, 100)
        # size = (25_000, 100_000)  # The max size we want to support

        with tempfile.TemporaryFile() as fX:
            with tempfile.NamedTemporaryFile() as fout:
                X = np.memmap(fX, "float64", mode="w+", shape=size)
                X[:, :10] = rng.uniform(size=(X.shape[0], 10))  # so we don't get stddev = 0
                X[:, 10:] = 1
                out = np.memmap(fout, "float64", mode="w+", shape=(X.shape[0], X.shape[0]))

                corrcoef(X, rowvar=True, out=out, copy=False)

    # the following tests are adapted from NumPy

    def test_non_array(self):
        np.testing.assert_almost_equal(
            corrcoef([[0, 1, 0], [1, 0, 1]]), [[1., -1.], [-1.,  1.]])

    def test_simple(self):
        A = np.array(
            [[0.15391142, 0.18045767, 0.14197213],
             [0.70461506, 0.96474128, 0.27906989],
             [0.9297531, 0.32296769, 0.19267156]])
        res1 = np.array(
            [[1., 0.9379533, -0.04931983],
             [0.9379533, 1., 0.30007991],
             [-0.04931983, 0.30007991, 1.]])
        tgt1 = corrcoef(A)
        np.testing.assert_almost_equal(tgt1, res1)
        self.assertTrue(np.all(np.abs(tgt1) <= 1.0))

    def test_complex(self):
        x = np.array([[1, 2, 3], [1j, 2j, 3j]])
        res = corrcoef(x)
        tgt = np.array([[1., -1.j], [1.j, 1.]])
        np.testing.assert_allclose(res, tgt)
        self.assertTrue(np.all(np.abs(res) <= 1.0))


class TestCov(unittest.TestCase):
    def test_agreement(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(size=(10, 20))
        for rowvar in (True, False):
            with self.subTest(rowvar=rowvar):
                np.testing.assert_array_equal(cov(X, rowvar=rowvar), np.cov(X, rowvar=rowvar))

    def test_memmap(self):
        # Smoketest for memmap.
        # here isn't really a nice way to test memory usage but it's useful to
        # have this test present for manual testing
        size = (1_000, 100)
        # size = (25_000, 100_000)  # The max size we want to support

        with tempfile.TemporaryFile() as fX:
            with tempfile.NamedTemporaryFile() as fout:
                X = np.memmap(fX, "float64", mode="w+", shape=size)
                X[:] = 1
                out = np.memmap(fout, "float64", mode="w+", shape=(X.shape[0], X.shape[0]))

                cov(X, rowvar=True, out=out, copy=False)

    # the following tests are adapted from NumPy

    def test_basic(self):
        x1 = np.array([[0, 2], [1, 1], [2, 0]]).T
        res1 = np.array([[1., -1.], [-1., 1.]])
        np.testing.assert_allclose(cov(x1), res1)

    def test_complex(self):
        x = np.array([[1, 2, 3], [1j, 2j, 3j]])
        res = np.array([[1., -1.j], [1.j, 1.]])
        np.testing.assert_allclose(cov(x), res)

    def test_1D_rowvar(self):
        x3 = np.array([0.3942, 0.5969, 0.7730, 0.9918, 0.7964])
        np.testing.assert_allclose(cov(x3), cov(x3, rowvar=False))


class TestDot2D(unittest.TestCase):
    def test_agreement(self):
        rng = np.random.default_rng(42)
        X = rng.uniform(size=(10, 20))
        Y = rng.uniform(size=(20, 100))
        np.testing.assert_array_equal(dot_2d(X, Y), np.dot(X, Y))

    def test_chunksize(self):
        # make sure that chunk sizes that don't align with the total number
        # of rows still work
        rng = np.random.default_rng(42)
        X = rng.uniform(size=(10, 20))
        Y = rng.uniform(size=(20, 15))
        np.testing.assert_array_almost_equal(dot_2d(X, Y, chunksize=86), np.dot(X, Y))
        np.testing.assert_array_almost_equal(dot_2d(X, Y, chunksize=365), np.dot(X, Y))

    def test_memmap(self):
        # Smoketest for memmap.
        # here isn't really a nice way to test memory usage but it's useful to
        # have this test present for manual testing
        size = (1_000, 100)
        # size = (25_000, 100_000)  # The max size we want to support

        with tempfile.TemporaryFile() as fX:
            with tempfile.NamedTemporaryFile() as fout:
                X = np.memmap(fX, "float64", mode="w+", shape=size)
                X[:] = 1
                out = np.memmap(fout, "float64", mode="w+", shape=(X.shape[0], X.shape[0]))

                dot_2d(X, X.T, out=out)


class TestMI(unittest.TestCase):
    def test_cmi_mixed(self):
        np.random.seed(42)
        n_samples, n_neighbors = 4003, 4
        c1 = np.random.randn(n_samples,)
        n_ss = 1001
        d1 = np.hstack((
            np.random.randint(-2, 1, (n_ss, )),
            np.random.randint(10, 14, (n_samples-n_ss, ))))
        d2 = np.random.randint(0, 4, (n_samples, ))
        np.random.shuffle(d2)
        cmi_ij_pair = _compute_cmip_c(c1, d1, d2, n_neighbors=n_neighbors)
        cmi_ij_pair_cdc = _compute_cmip_cdc(c1, d1, d2, n_neighbors=n_neighbors)
        cmi_ij_pair_ccd = _compute_cmip_ccd(c1, d1, d2, n_neighbors=n_neighbors)
        self.assertAlmostEqual(
            sum(cmi_ij_pair), sum(cmi_ij_pair_cdc), places=4,
            msg="Computation for mixed continuous/discrete conditional mutual information is not accurate")
        self.assertAlmostEqual(
            sum(cmi_ij_pair), sum(cmi_ij_pair_ccd), places=4,
            msg="Computation for mixed continuous/discrete conditional mutual information is not accurate")

    def test_cmi_symmetry(self):
        np.random.seed(42)
        n_samples, n_neighbors = 4003, 4
        # Test continuous implementation
        Xy = np.random.randn(n_samples, 3)
        cmi_ij_pair = _compute_cmip_c(Xy[:, 0], Xy[:, 1], Xy[:, 2], n_neighbors=n_neighbors)
        cmi_ji_pair = _compute_cmip_c(Xy[:, 1], Xy[:, 0], Xy[:, 2], n_neighbors=n_neighbors)
        self.assertAlmostEqual(
            sum(cmi_ij_pair), sum(cmi_ji_pair), places=3,
            msg="Computation for continuous conditional mutual information is not symmetric")

        c1 = np.random.randn(n_samples,)
        c2 = np.random.randn(n_samples,)
        n_ss = 1001
        d = np.hstack((
            np.random.randint(-2, 1, (n_ss, )),
            np.random.randint(10, 14, (n_samples-n_ss, ))))
        np.random.shuffle(d)
        cmi_ij_pair = _compute_cmip_ccd(c1, c2, d, n_neighbors=n_neighbors)
        cmi_ji_pair = _compute_cmip_ccd(c2, c1, d, n_neighbors=n_neighbors)
        self.assertAlmostEqual(
            sum(cmi_ij_pair), sum(cmi_ji_pair), places=3,
            msg="Computation for mixed continuous/discrete conditional mutual information is not symmetric")

    def test_cmi_discrete(self):
        """
        We test the algorithm using the formula `I(x;y|z) = I(x;y) + I(z;y) - I(z;y|x)`.
        Since the mutual information data is an sklearn function,
        :func:`sklearn.feature_selection._mutual_info.mutual_info_classif`
        it is highly likely that formula is valid only if our algorithm is correct.
        """
        np.random.seed(42)
        # Test discrete implementation
        n_samples, n_ss, n_neighbors = 103, 51, 4
        xi = np.random.randint(0, 11, (n_samples, ))
        xj = np.hstack((
            np.random.randint(-2, 1, (n_ss, )),
            np.random.randint(10, 14, (n_samples-n_ss, ))))
        np.random.shuffle(xi)
        np.random.shuffle(xj)
        y = np.random.randint(-12, -10, (n_samples, ))
        cmi_ij, cmi_ji = _compute_cmip_d(xi, xj, y)
        mi_i = mutual_info_classif(
            xi.reshape(-1, 1), y, discrete_features=True, n_neighbors=n_neighbors)
        mi_j = mutual_info_classif(
            xj.reshape(-1, 1), y, discrete_features=True, n_neighbors=n_neighbors)
        self.assertTrue(
            np.allclose(cmi_ij + mi_j - mi_i, cmi_ji, atol=1e-5),
            msg="The chain rule for discrete conditional mutual information is violated")
        cmi_ij2 = _compute_cmip_d(xj, xi, y)
        self.assertTrue(
            np.allclose(sum(cmi_ij2), cmi_ji+cmi_ij, atol=1e-5),
            msg="Discrete conditional mutual information computation is not symmetric")

    def test_mi(self):
        from sklearn.feature_selection._mutual_info import _compute_mi_cd, _compute_mi_cc
        from sklearn.metrics.cluster import mutual_info_score
        # Test discrete implementation
        n_samples, n_ss, n_neighbors, random_state = 103, 51, 4, 15
        np.random.seed(random_state)
        xi = np.random.randint(0, 11, (n_samples, ))
        xj = np.hstack((
            np.random.randint(-2, 1, (n_ss, )),
            np.random.randint(10, 14, (n_samples-n_ss, ))))
        np.random.shuffle(xi)
        np.random.shuffle(xj)
        xi_c = scale(np.random.randn(n_samples), with_mean=False)
        xj_c = scale(np.random.randn(n_samples), with_mean=False)

        mi_ij = _compute_mi(
            xi, xj, True, True, n_neighbors=n_neighbors)
        mi_ij_cl = mutual_info_score(xi, xj)
        self.assertTrue(
            np.allclose(mi_ij_cl, mi_ij),
            msg="Discrete mutual information is not computed correctly")

        mi_ij = _compute_mi(
            xi_c, xj, False, True, n_neighbors=n_neighbors)
        mi_ij_cl = _compute_mi_cd(xi_c, xj, n_neighbors=n_neighbors)
        self.assertTrue(
            np.allclose(mi_ij_cl, mi_ij, atol=1e-5),
            msg=f"Error in continuous features/discrete target MI estimation is larger than expected {abs(mi_ij_cl-mi_ij)}")

        mi_ij = _compute_mi(xi_c, xj_c, False, False, n_neighbors=n_neighbors)
        mi_ij_cl = _compute_mi_cc(xi_c, xj_c, n_neighbors=n_neighbors)
        self.assertTrue(
            np.allclose(mi_ij_cl, mi_ij, atol=1e-5),
            msg=f"Error in purely continuous MI is larger than expected {abs(mi_ij_cl-mi_ij)}")

    def test_cmi_continuous(self):
        # methods from https://github.com/jannisteunissen/mutual_information
        def _compute_cmi_t(x, z, y, n_neighbors):
            n_samples = len(x)

            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            z = z.reshape(-1, 1)
            xyz = np.hstack((x, y, z))
            k = np.full(n_samples, n_neighbors)
            radius = get_radius_kneighbors(xyz, n_neighbors)

            mask = (radius == 0)
            if mask.sum() > 0:
                vals, ix, counts = np.unique(xyz[mask], axis=0,
                                             return_inverse=True,
                                             return_counts=True)
                k[mask] = counts[ix] - 1

            nxz = num_points_within_radius(np.hstack((x, z)), radius)
            nyz = num_points_within_radius(np.hstack((y, z)), radius)
            nz = num_points_within_radius(z, radius)

            cmi = max(0, np.mean(digamma(k)) - np.mean(digamma(nxz + 1))
                      - np.mean(digamma(nyz + 1)) + np.mean(digamma(nz + 1)))
            return cmi

        def get_radius_kneighbors(x, n_neighbors):
            """Determine smallest radius around x containing n_neighbors neighbors

            :param x: ndarray, shape (n_samples, n_dim)
            :param n_neighbors: number of neighbors
            :returns: radius, shape (n_samples,)
            """
            # Use KDTree for simplicity (sometimes a ball tree could be faster)
            kd = KDTree(x, metric="chebyshev")

            # Results include point itself, therefore n_neighbors+1
            neigh_dist = kd.query(x, k=n_neighbors+1)[0]

            # Take radius slightly larger than distance to last neighbor
            radius = np.nextafter(neigh_dist[:, -1], 0)
            return radius

        def num_points_within_radius(x, radius):
            """For each point, determine the number of other points within a given radius

            :param x: ndarray, shape (n_samples, n_dim)
            :param radius: radius, shape (n_samples,)
            :returns: number of points within radius
            """
            kd = KDTree(x, metric="chebyshev")
            nx = kd.query_radius(x, radius, count_only=True, return_distance=False)
            return np.array(nx) - 1.0

        np.random.seed(42)
        n_samples, n_neighbors = 103, 4
        # Test continuous implementation
        Xy = np.random.randn(n_samples, 3)
        cmi_t_01 = _compute_cmi_t(Xy[:, 0], Xy[:, 1], Xy[:, 2], n_neighbors=n_neighbors)
        cmi_t_10 = _compute_cmi_t(Xy[:, 1], Xy[:, 0], Xy[:, 2], n_neighbors=n_neighbors)

        cmi_ij, cmi_ji = _compute_cmip_c(Xy[:, 0], Xy[:, 1], Xy[:, 2], n_neighbors=n_neighbors)

        self.assertAlmostEqual(
            max(cmi_t_01, 0), cmi_ij, places=5,
            msg="The algorithm doesn't match the original implementation")

        self.assertAlmostEqual(
            max(cmi_t_10, 0), cmi_ji, places=5,
            msg="The algorithm doesn't match the original implementation")
