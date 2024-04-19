
# Tests in TestMI are adapted from sklearn tests and licenced under:
# BSD 3-Clause License
#
# Copyright (c) 2007-2024 The scikit-learn developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
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
# Tests in TestCMI are adapted from https://github.com/jannisteunissen/mutual_information and
# distributed under the following licence:
#
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
# The modifications of the test and the tests in TestDCMI are distributed
# under the Apache 2.0 Software license:
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

import unittest

import numpy as np

from dwave.plugins.sklearn._conditional_mutual_info import (
    _compute_mi,
    _compute_cmip_c, _compute_cmip_d,
    _compute_cmip_cdc, _compute_cmip_ccd
)

from numpy.linalg import det
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state


class TestMI(unittest.TestCase):
    def test_against_sklearn(self):
        # the test will break if sklearn modifies its implementation
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

    def test_compute_mi_dd(self):
        # In discrete case computations are straightforward and can be done
        # by hand on given vectors.
        x = np.array([0, 1, 1, 0, 0])
        y = np.array([1, 0, 0, 0, 1])

        H_x = H_y = -(3 / 5) * np.log(3 / 5) - (2 / 5) * np.log(2 / 5)
        H_xy = -1 / 5 * np.log(1 / 5) - 2 / 5 * np.log(2 / 5) - 2 / 5 * np.log(2 / 5)
        I_xy = H_x + H_y - H_xy

        self.assertAlmostEqual(_compute_mi(x, y, discrete_feature_x=True,
                               discrete_feature_y=True), I_xy, places=4)

    def test_compute_mi_cc(self):
        # For two continuous variables a good approach is to test on bivariate
        # normal distribution, where mutual information is known.

        # Mean of the distribution, irrelevant for mutual information.
        mean = np.zeros(2)

        # Setup covariance matrix with correlation coeff. equal 0.5.
        sigma_1 = 1
        sigma_2 = 10
        corr = 0.5
        cov = np.array(
            [
                [sigma_1**2, corr * sigma_1 * sigma_2],
                [corr * sigma_1 * sigma_2, sigma_2**2],
            ]
        )

        # True theoretical mutual information.
        I_theory = np.log(sigma_1) + np.log(sigma_2) - 0.5 * np.log(np.linalg.det(cov))

        rng = check_random_state(0)
        Z = rng.multivariate_normal(mean, cov, size=5001).astype(np.float64, copy=False)

        x, y = Z[:, 0], Z[:, 1]

        # Theory and computed values won't be very close
        # We here check with a large relative tolerance
        for n_neighbors in [4, 6, 8]:
            I_computed = _compute_mi(
                x, y, discrete_feature_x=False, discrete_feature_y=False, n_neighbors=n_neighbors
            )
            self.assertLessEqual(abs(I_computed/I_theory-1.0), 1e-1)

    def test_compute_mi_cd(self):
        # To test define a joint distribution as follows:
        # p(x, y) = p(x) p(y | x)
        # X ~ Bernoulli(p)
        # (Y | x = 0) ~ Uniform(-1, 1)
        # (Y | x = 1) ~ Uniform(0, 2)

        # Use the following formula for mutual information:
        # I(X; Y) = H(Y) - H(Y | X)
        # Two entropies can be computed by hand:
        # H(Y) = -(1-p)/2 * ln((1-p)/2) - p/2*log(p/2) - 1/2*log(1/2)
        # H(Y | X) = ln(2)

        # Now we need to implement sampling from out distribution, which is
        # done easily using conditional distribution logic.

        n_samples = 5001
        rng = check_random_state(0)

        for p in [0.3, 0.5, 0.7]:
            x = rng.uniform(size=n_samples) > p

            y = np.empty(n_samples, np.float64)
            mask = x == 0
            y[mask] = rng.uniform(-1, 1, size=np.sum(mask))
            y[~mask] = rng.uniform(0, 2, size=np.sum(~mask))

            I_theory = -0.5 * (
                (1 - p) * np.log(0.5 * (1 - p)) + p * np.log(0.5 * p) + np.log(0.5)
            ) - np.log(2)

            # Assert the same tolerance.
            for n_neighbors in [4, 6, 8]:
                I_computed = _compute_mi(
                    x, y, discrete_feature_x=True, discrete_feature_y=False, n_neighbors=n_neighbors
                )
                self.assertLessEqual(abs(I_computed/I_theory-1.0), 1e-1)

    def test_compute_mi_cd_unique_label(self):
        # Test that adding unique label doesn't change MI.
        n_samples = 100
        x = np.random.uniform(size=n_samples) > 0.5

        y = np.empty(n_samples, np.float64)
        mask = x == 0
        y[mask] = np.random.uniform(-1, 1, size=np.sum(mask))
        y[~mask] = np.random.uniform(0, 2, size=np.sum(~mask))

        mi_1 = _compute_mi(x, y, discrete_feature_x=True, discrete_feature_y=False)

        x = np.hstack((x, 2))
        y = np.hstack((y, 10))
        mi_2 = _compute_mi(x, y, discrete_feature_x=True, discrete_feature_y=False)

        self.assertAlmostEqual(mi_1, mi_2, places=5)


class TestCMI(unittest.TestCase):
    def test_discrete(self):
        # Third test from "Conditional Mutual Information Estimation for Mixed Discrete
        # and Continuous Variables with Nearest Neighbors" (Mesner & Shalizi). Fully
        # discrete
        def rescale(xi):
            xi = scale(xi + 0.0, with_mean=False, copy=False)
            # Add small noise to continuous features as advised in Kraskov et. al.
            xi = xi.astype(np.float64, copy=False)
            means = np.maximum(1, np.mean(np.abs(xi), axis=0))
            xi += (1e-10 * means * rng.standard_normal(size=(xi.shape[0], )))
            return xi
        N = 1001
        rng = check_random_state(0)
        choices = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
        p_discrete = rng.choice(4, p=[0.4, 0.4, 0.1, 0.1], size=N)
        xy_discrete = choices[p_discrete]
        x, y = xy_discrete[:, 0], xy_discrete[:, 1]
        z = rng.poisson(2, size=N)
        x_c = rescale(x)
        y_c = rescale(y)
        z_c = rescale(z)
        cmi_analytic = 2 * 0.4 * np.log(0.4/0.5**2) + 2 * 0.1 * np.log(0.1/0.5**2)
        # checking the computations with a large relative tolerance
        for n_neighbors in [3, 5, 7]:
            cmi_ij, _ = _compute_cmip_d(x, z, y)
            self.assertLessEqual(abs(cmi_ij/cmi_analytic-1.0), 1e-1)
            cmi_ij, _ = _compute_cmip_c(x_c, z_c, y_c, n_neighbors)
            self.assertLessEqual(abs(cmi_ij/cmi_analytic-1.0), 1e-1)
            cmi_ij, _ = _compute_cmip_ccd(x_c, z_c, y, n_neighbors)
            self.assertLessEqual(abs(cmi_ij/cmi_analytic-1.0), 1e-1)
            cmi_ij, _ = _compute_cmip_cdc(x_c, z, y_c, n_neighbors)
            self.assertLessEqual(abs(cmi_ij/cmi_analytic-1.0), 1e-1)

    def test_bivariate(self):
        """Test with bivariate normal variables"""
        N = 1001
        rng = check_random_state(0)
        mu = np.zeros(2)
        cov = np.array([[1., 0.8], [0.8, 1.0]])
        xy_gauss = rng.multivariate_normal(mu, cov, size=N)
        x, y = xy_gauss[:, 0], xy_gauss[:, 1]
        z = rng.normal(size=N)

        cmi_analytic = -0.5 * np.log(det(cov))
        # checking the computations with a large relative tolerance
        for n_neighbors in [4, 6, 8]:
            cmi_ij, _ = _compute_cmip_c(x, z, y, n_neighbors)
            self.assertAlmostEqual(cmi_ij/cmi_analytic, 1.0, places=1)

    def test_trivariate(self):
        # Test with 'trivariate' normal variables x, y, z

        mu = np.zeros(3)

        # Covariance matrix
        cov_xy = 0.7
        cov_xz = 0.5
        cov_yz = 0.3
        cov = np.array([[1, cov_xy, cov_xz],
                        [cov_xy, 1.0, cov_yz],
                        [cov_xz, cov_yz, 1]])
        N = 1001
        rng = check_random_state(0)
        samples = rng.multivariate_normal(mu, cov, size=N)
        x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]

        # Construct minor matrices for x and y
        cov_x = cov[1:, 1:]
        cov_y = cov[[0, 2]][:, [0, 2]]
        cmi_analytic = -0.5 * np.log(det(cov) / (det(cov_x) * det(cov_y)))
        # checking the computations with a large relative tolerance
        for n_neighbors in [4, 6, 8]:
            cmi_ij, _ = _compute_cmip_c(x, z, y, n_neighbors)
            self.assertAlmostEqual(cmi_ij/cmi_analytic, 1, places=1)


class TestDCMI(unittest.TestCase):
    def test_cmi_symmetry(self):
        # The pair produced by the computations should be symmetric in the first two arguments
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
        # We test the algorithm using the formula `I(x;y|z) = I(x;y) + I(z;y) - I(z;y|x)`.
        # It is highly likely if our algorithm is correct, then formula holds.
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
        # we estimate mutual information using
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
