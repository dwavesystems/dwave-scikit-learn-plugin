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

from dwave.plugins.sklearn.utilities import corrcoef, cov, dot_2d


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
