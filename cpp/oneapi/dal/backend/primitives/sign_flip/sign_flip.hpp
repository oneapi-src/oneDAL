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

#ifdef ONEDAL_DATA_PARALLEL

/// Computes correlation matrix from covariance matrix
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  row_count  The number of rows
/// @param[in]  sums  The [p] sums computed along each column of the data
/// @param[out] cov   The [p x p] covariance matrix
/// @param[out] corr  The [p x p] correlation matrix
/// @param[out] tmp   The [p] temporary buffer
template <typename Float>
sycl::event sign_flip(sycl::queue q, ndview<Float, 2>& eigvecs, const event_vector& deps = {});

#endif

} // namespace oneapi::dal::backend::primitives
