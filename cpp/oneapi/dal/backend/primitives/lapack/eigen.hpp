/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

/// Do not use this.
template <typename Float>
void sym_eigval_impl(Float* a, std::int64_t n, std::int64_t lda, Float* w);

/// Computes eigenvectors and eigenvalues inplace.
///
/// @param[in, out] data_or_eigenvectors The input parameter is interpreted as symmetric matrix of
///                                      size [n x n]. The computed eigenvectors is written to that
///                                      matrix. If `order == ndorder::c`, $i$-th row of the matrix
///                                      contains $i$-th eigenvector. If `order == ndorder::f`, $i$-th
///                                      column of the matrix contains $i$-th eigenvector.
/// @param[out] eigenvalues              The output array of size [n] that stores computed eigenvalues.
///                                      The eigenvalues are written in ascending order. $i$-th eigenvalue
///                                      corrensponds to $i$-th eigenvector.
template <typename Float, ndorder order>
void sym_eigval(ndview<Float, 2, order>& data_or_eigenvectors, ndview<Float, 1>& eigenvalues) {
    ONEDAL_ASSERT(data_or_eigenvectors.get_dimension(0) == data_or_eigenvectors.get_dimension(1),
                  "Input matrix must be square");
    ONEDAL_ASSERT(eigenvalues.get_dimension(0) >= data_or_eigenvectors.get_dimension(0));
    ONEDAL_ASSERT(data_or_eigenvectors.has_mutable_data());
    ONEDAL_ASSERT(eigenvalues.has_mutable_data());

    sym_eigval_impl(data_or_eigenvectors.get_mutable_data(),
                    data_or_eigenvectors.get_dimension(0),
                    data_or_eigenvectors.get_leading_stride(),
                    eigenvalues.get_mutable_data());
}

} // namespace oneapi::dal::backend::primitives
