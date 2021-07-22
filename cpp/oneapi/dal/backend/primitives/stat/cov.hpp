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

/// Computes correlation matrix and variances
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  data  The [n x p] input dataset
/// @param[in]  sums  The [p] sums computed along each column of the data
/// @param[out] corr  The [p x p] correlation matrix
/// @param[out] means The [p] means for each feature
/// @param[out] vars  The [p] variances for each feature
/// @param[out] tmp   The [p] temporary buffer
template <typename Float>
sycl::event correlation(sycl::queue& queue,
                        const ndview<Float, 2>& data,
                        const ndview<Float, 1>& sums,
                        ndview<Float, 2>& corr,
                        ndview<Float, 1>& means,
                        ndview<Float, 1>& vars,
                        ndview<Float, 1>& tmp,
                        const event_vector& deps = {});

#endif

} // namespace oneapi::dal::backend::primitives
