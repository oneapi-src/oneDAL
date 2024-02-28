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

/// Compute means
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  row_count  The number of rows
/// @param[in]  sums  The [p] sums computed along each column of the data
/// @param[out] means The [p] means for each feature
template <typename Float>
sycl::event means(sycl::queue& queue,
                  std::int64_t row_count,
                  const ndview<Float, 1>& sums,
                  ndview<Float, 1>& means,
                  const event_vector& deps = {});

/// Computes covariance matrix
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  row_count  The number of rows
/// @param[in]  sums  The [p] sums computed along each column of the data
/// @param[in]  bias If true biased covariance estimated by maximum likelihood method computed
/// @param[out] cov  The [p x p] covariance matrix
template <typename Float>
sycl::event covariance(sycl::queue& q,
                       std::int64_t row_count,
                       const ndview<Float, 1>& sums,
                       ndview<Float, 2>& cov,
                       bool bias,
                       const event_vector& deps = {});

/// Compute variances
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  row_count  The number of rows
/// @param[in]  cov  The [p x p] covariance matrix
/// @param[in]  sums  The [p] sums computed along each column of the data
/// @param[out] vars The [p] vars for each feature
template <typename Float>
sycl::event variances(sycl::queue& queue,
                      const ndview<Float, 2>& cov,
                      ndview<Float, 1>& vars,
                      const event_vector& deps = {});

/// Computes correlation matrix
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  row_count  The number of rows
/// @param[in]  sums  The [p] sums computed along each column of the data
/// @param[out] corr  The [p x p] correlation matrix
/// @param[out] tmp   The [p] temporary buffer
template <typename Float>
sycl::event correlation(sycl::queue& q,
                        std::int64_t row_count,
                        const ndview<Float, 1>& sums,
                        ndview<Float, 2>& corr,
                        const event_vector& deps = {});

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
/// @param[in]  bias  Determines if provided covariance estimation biased
template <typename Float>
sycl::event correlation_from_covariance(sycl::queue& q,
                                        std::int64_t row_count,
                                        const ndview<Float, 2>& cov,
                                        ndview<Float, 2>& corr,
                                        bool bias,
                                        const event_vector& deps = {});

#endif

} // namespace oneapi::dal::backend::primitives
