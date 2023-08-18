/*******************************************************************************
* Copyright 2023 Intel Corporation
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

template <typename Float, typename BinaryOp, typename UnaryOp>
Float reduce_1d_impl(sycl::queue& q,
                     const ndview<Float, 1>& input,
                     const BinaryOp& binary = {},
                     const UnaryOp& unary = {},
                     const event_vector& deps = {});

template <typename Float, typename BinaryOp, typename UnaryOp>
sycl::event reduce_1d_impl(sycl::queue& q,
                           const ndview<Float, 1>& input,
                           ndview<Float, 1>& output,
                           const BinaryOp& binary = {},
                           const UnaryOp& unary = {},
                           const event_vector& deps = {});

/// Reduces `input` rows and returns the result
///
/// @tparam Float    Floating-point type used to perform computations
/// @tparam BinaryOp Type of binary operator functor
/// @tparam UnaryOp  Type of unary operator functor
///
/// @param[in]  queue   The SYCL queue
/// @param[in]  input   The [n] input array
/// @param[in]  binary  The binary functor that reduces two values into one
/// @param[in]  unary   The unary functor that performs element-wise operation before reduction
/// @param[in]  deps    The vector of `sycl::event`s that represents list of dependencies
template <typename Float, typename BinaryOp, typename UnaryOp>
inline Float reduce_1d(sycl::queue& q,
                       const ndview<Float, 1>& input,
                       const BinaryOp& binary = {},
                       const UnaryOp& unary = {},
                       const event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(reduction.reduce_1d, q);
    static_assert(dal::detail::is_tag_one_of_v<BinaryOp, reduce_binary_op_tag>,
                  "BinaryOp must be a special binary operation defined "
                  "at the primitives level");
    static_assert(dal::detail::is_tag_one_of_v<UnaryOp, reduce_unary_op_tag>,
                  "UnaryOp must be a special unary operation defined "
                  "at the primitives level");
    return reduce_1d_impl(q, input, binary, unary, deps);
}

/// Reduces `input` rows and put the result in output
///
/// @tparam Float    Floating-point type used to perform computations
/// @tparam BinaryOp Type of binary operator functor
/// @tparam UnaryOp  Type of unary operator functor
///
/// @param[in]  queue   The SYCL queue
/// @param[in]  input   The [n] input array
/// @param[out] output  The [1] array where the output is put
/// @param[in]  binary  The binary functor that reduces two values into one
/// @param[in]  unary   The unary functor that performs element-wise operation before reduction
/// @param[in]  deps    The vector of `sycl::event`s that represents list of dependencies
template <typename Float, typename BinaryOp, typename UnaryOp>
inline sycl::event reduce_1d(sycl::queue& q,
                             const ndview<Float, 1>& input,
                             ndview<Float, 1>& output,
                             const BinaryOp& binary = {},
                             const UnaryOp& unary = {},
                             const event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(reduction.reduce_1d, q);
    static_assert(dal::detail::is_tag_one_of_v<BinaryOp, reduce_binary_op_tag>,
                  "BinaryOp must be a special binary operation defined "
                  "at the primitives level");
    static_assert(dal::detail::is_tag_one_of_v<UnaryOp, reduce_unary_op_tag>,
                  "UnaryOp must be a special unary operation defined "
                  "at the primitives level");
    return reduce_1d_impl(q, input, output, binary, unary, deps);
}

#endif

} // namespace oneapi::dal::backend::primitives
