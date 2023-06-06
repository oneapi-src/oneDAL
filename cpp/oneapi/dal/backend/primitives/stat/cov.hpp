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

/// Subtract 1-d array from 2-d array
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  row_count  The number of rows
/// @param[in]  minuend  A 2-d array, from which another array will be subtracted
/// @param[in]  subtrahend  A 1-d array, which is to be subtracted from minuend
/// @param[out] difference The difference between minuend and
template <typename Float>
sycl::event elementwise_difference(sycl::queue& queue,
                                   std::int64_t row_count,
                                   const ndview<Float, 2>& minuend,
                                   const ndview<Float, 1>& subtrahend,
                                   ndview<Float, 2>& difference,
                                   const event_vector& deps = {});

/// SQRT of 1-d array
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  src  source array
/// @param[out]  dst  destination array with elementwise square-root of source array
template <typename Float>
sycl::event elementwise_sqrt(sycl::queue& queue,
                             const ndview<Float, 1>& src,
                             ndview<Float, 1>& dst,
                             const event_vector& deps = {});

/// Divide 2-d array by 1-d array
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  row_count  The number of rows
/// @param[in]  numerator  A 2-d array
/// @param[in]  denominator  A 1-d array
/// @param[out] quotient The result of the division
template <typename Float>
sycl::event elementwise_division(sycl::queue& queue,
                                 std::int64_t row_count,
                                 const ndview<Float, 2>& numerator,
                                 const ndview<Float, 1>& denominator,
                                 ndview<Float, 2>& quotient,
                                 const event_vector& deps = {});

/// Computes covariance matrix
///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  queue The queue
/// @param[in]  row_count  The number of rows
/// @param[in]  sums  The [p] sums computed along each column of the data
/// @param[out] cov  The [p x p] covariance matrix
template <typename Float>
sycl::event covariance(sycl::queue& q,
                       std::int64_t row_count,
                       const ndview<Float, 1>& sums,
                       ndview<Float, 2>& cov,
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
                        ndview<Float, 1>& tmp,
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
template <typename Float>
sycl::event correlation_from_covariance(sycl::queue& q,
                                        std::int64_t row_count,
                                        const ndview<Float, 2>& cov,
                                        ndview<Float, 2>& corr,
                                        ndview<Float, 1>& tmp,
                                        const event_vector& deps = {});

#endif

} // namespace oneapi::dal::backend::primitives
