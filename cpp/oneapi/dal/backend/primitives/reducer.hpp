/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <cmath>
#include <cstdint>

#ifdef ONEAPI_DAL_DATA_PARALLEL
    #include <CL/sycl.hpp>
#endif

#include "oneapi/dal/data/array.hpp"

namespace oneapi::dal::backend::primitives {

enum class unary_operation : int { identity, square, abs };

template <typename Float, unary_operation Op = unary_operation::identity>
struct unary_functor {
    constexpr static Float operator()(Float arg) const;
};

enum class binary_operation : int { min, max, sum, mul };

template <typename Float, binary_operation Op>
struct binary_functor {
    //should be initialized anyway
    constexpr static Float init_value = static_cast<Float>(NAN);
    constexpr static Float operator()(Float a, Float b);
};

#ifdef ONEAPI_DAL_DATA_PARALLEL

template <unary_operation UnOp,
          binary_operation BinOp,
          typename Float,
          bool IsRowMajorLayout = true>
struct reducer_singlepass {
    reducer_singlepass(cl::sycl::queue& q);
    cl::sycl::event operator()(array<Float> input,
                               array<Float> output,
                               std::int64_t vector_size,
                               std::int64_t n_vectors,
                               std::int64_t work_items_per_group);
    cl::sycl::event operator()(array<Float> input,
                               array<Float> output,
                               std::int64_t vector_size,
                               std::int64_t n_vectors);
    typename array<Float> operator()(array<Float> input,
                                     std::int64_t vector_size,
                                     std::int64_t n_vectors);

private:
    cl::sycl::queue& _q;
};

template <typename Float>
using l1_reducer_singlepass =
    typename reducer_singlepass<unary_operation::abs, binary_operation::sum, Float>;
template <typename Float>
using l2_reducer_singlepass =
    typename reducer_singlepass<unary_operation::square, binary_operation::sum, Float>;
template <typename Float>
using linf_reducer_singlepass =
    typename reducer_singlepass<unary_operation::abs, binary_operation::max, Float>;

#endif

} // namespace oneapi::dal::backend::primitives
