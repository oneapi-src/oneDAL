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

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/reduction/functors.hpp"
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float, ndorder order, typename BinaryOp, typename UnaryOp>
sycl::event reduce_by_rows_impl(sycl::queue& q,
                                const ndview<Float, 2, order>& input,
                                ndview<Float, 1>& output,
                                const BinaryOp& binary,
                                const UnaryOp& unary,
                                const event_vector& deps,
                                bool override_init = true);

/// Reduces `input` rows and stores results into `output`
///
/// @tparam Float    Floating-point type used to perform computations
/// @tparam order    Input matrix data layout
/// @tparam BinaryOp Type of binary operator functor
/// @tparam UnaryOp  Type of unary operator functor
///
/// @param[in]  queue         The SYCL queue
/// @param[in]  input         The [n x p] input dataset
/// @param[out] output        The [n] results of reduction
/// @param[in]  binary        The binary functor that reduces two values into one
/// @param[in]  unary         The unary function that performs the element-wise operation before reduction
/// @param[in]  deps          The vector of `sycl::event` that represents a list of dependencies
/// @param[in]  override_init Should the value stored in output be used in reduction or be overwritten
template <typename Float, ndorder order, typename BinaryOp, typename UnaryOp>
inline sycl::event reduce_by_rows(sycl::queue& q,
                                  const ndview<Float, 2, order>& input,
                                  ndview<Float, 1>& output,
                                  const BinaryOp& binary = BinaryOp{},
                                  const UnaryOp& unary = UnaryOp{},
                                  const event_vector& deps = {},
                                  bool override_init = true) {
    ONEDAL_PROFILER_TASK(reduction.reduce_by_rows, q);
    static_assert(dal::detail::is_tag_one_of_v<BinaryOp, reduce_binary_op_tag>,
                  "BinaryOp must be a special binary operation defined "
                  "at the primitives level");
    static_assert(dal::detail::is_tag_one_of_v<UnaryOp, reduce_unary_op_tag>,
                  "UnaryOp must be a special unary operation defined "
                  "at the primitives level");
    return reduce_by_rows_impl(q, input, output, binary, unary, deps, override_init);
}

template <typename Float, ndorder order, typename BinaryOp, typename UnaryOp>
sycl::event reduce_by_columns_impl(sycl::queue& q,
                                   const ndview<Float, 2, order>& input,
                                   ndview<Float, 1>& output,
                                   const BinaryOp& binary,
                                   const UnaryOp& unary,
                                   const event_vector& deps,
                                   bool override_init = true);

/// Reduces `input` columns and stores results into `output`
///
/// @tparam Float    Floating-point type used to perform computations
/// @tparam order    Input matrix data layout
/// @tparam BinaryOp Type of binary operator functor
/// @tparam UnaryOp  Type of unary operator functor
///
/// @param[in]  queue         The SYCL queue
/// @param[in]  input         The [n x p] input dataset
/// @param[out] output        The [p] results of reduction
/// @param[in]  binary        The binary functor that reduces two values to one
/// @param[in]  unary         The unary functor that performs the element-wise operation before reduction
/// @param[in]  deps          The vector of `sycl::event` that represents a list of dependencies
/// @param[in]  override_init Should the value stored in output be used in reduction or overwriten
template <typename Float, ndorder order, typename BinaryOp, typename UnaryOp>
inline sycl::event reduce_by_columns(sycl::queue& q,
                                     const ndview<Float, 2, order>& input,
                                     ndview<Float, 1>& output,
                                     const BinaryOp& binary = BinaryOp{},
                                     const UnaryOp& unary = UnaryOp{},
                                     const event_vector& deps = {},
                                     bool override_init = true) {
    ONEDAL_PROFILER_TASK(reduction.reduce_by_columns, q);
    static_assert(dal::detail::is_tag_one_of_v<BinaryOp, reduce_binary_op_tag>,
                  "BinaryOp must be a special binary operation defined "
                  "at the primitives level");
    static_assert(dal::detail::is_tag_one_of_v<UnaryOp, reduce_unary_op_tag>,
                  "UnaryOp must be a special unary operation defined "
                  "at the primitives level");
    return reduce_by_columns_impl(q, input, output, binary, unary, deps, override_init);
}

#endif

} // namespace oneapi::dal::backend::primitives
