
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

#include "oneapi/dal/array.hpp"

#include "oneapi/dal/backend/common.hpp"

#include "oneapi/dal/backend/primitives/functors.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEAPI_DAL_DATA_PARALLEL

namespace impl {

template <typename UnOp, typename BinOp, typename Float, bool VectorsAreRows>
struct reducer_singlepass_kernel;

} // namespace impl

template <typename Float,
          typename BinaryFunctor,
          typename UnaryFunctor = unary_functor<Float, unary_operation::identity>,
          layout Layout         = layout::row_major>
struct reducer_singlepass {
public:
    constexpr static inline bool vectors_are_rows = (Layout == layout::row_major);

private:
    typedef impl::reducer_singlepass_kernel<UnaryFunctor, BinaryFunctor, Float, vectors_are_rows>
        kernel_t;

public:
    reducer_singlepass(cl::sycl::queue& q,
                       BinaryFunctor binary_func = BinaryFunctor{},
                       UnaryFunctor unary_func   = UnaryFunctor{});
    cl::sycl::event operator()(array<Float> input,
                               array<Float> output,
                               std::int64_t vector_size,
                               std::int64_t n_vectors,
                               std::int64_t work_items_per_group);
    cl::sycl::event operator()(array<Float> input,
                               array<Float> output,
                               std::int64_t vector_size,
                               std::int64_t n_vectors);
    cl::sycl::event operator()(const Float* input,
                               Float* output,
                               std::int64_t vector_size,
                               std::int64_t n_vectors,
                               std::int64_t work_items_per_group);
    cl::sycl::event operator()(const Float* input,
                               Float* output,
                               std::int64_t vector_size,
                               std::int64_t n_vectors);

public:
    const BinaryFunctor binary;
    const UnaryFunctor unary;

private:
    cl::sycl::queue& q;

public:
    const std::int64_t max_work_group_size;
};

template <typename Float = float, layout Layout = layout::row_major>
using l2_reducer_singlepass = reducer_singlepass<Float,
                                                 binary_functor<Float, binary_operation::sum>,
                                                 unary_functor<Float, unary_operation::square>,
                                                 Layout>;

template <typename Float = float, layout Layout = layout::row_major>
using sum_reducer_singlepass = reducer_singlepass<Float,
                                                  binary_functor<Float, binary_operation::sum>,
                                                  unary_functor<Float, unary_operation::identity>,
                                                  Layout>;

template <typename Float,
          typename BinaryFunctor,
          typename UnaryFunctor = unary_functor<Float, unary_operation::identity>>
inline cl::sycl::event reduce_rm(cl::sycl::queue& q,
                                 const Float* input,
                                 Float* output,
                                 std::int64_t vector_size,
                                 std::int64_t n_vectors,
                                 BinaryFunctor binary_func         = BinaryFunctor{},
                                 UnaryFunctor unary_func           = UnaryFunctor{},
                                 std::int64_t work_items_per_group = -1) {
    reducer_singlepass<Float, BinaryFunctor, UnaryFunctor, layout::row_major> kernel(q,
                                                                                     binary_func,
                                                                                     unary_func);
    return (work_items_per_group <= 0)
               ? kernel(input, output, vector_size, n_vectors)
               : kernel(input, output, vector_size, n_vectors, work_items_per_group);
}

template <typename Float,
          typename BinaryFunctor,
          typename UnaryFunctor = unary_functor<Float, unary_operation::identity>>
inline cl::sycl::event reduce_rm(cl::sycl::queue& q,
                                 array<Float> input,
                                 array<Float> output,
                                 std::int64_t vector_size,
                                 std::int64_t n_vectors,
                                 BinaryFunctor binary_func         = BinaryFunctor{},
                                 UnaryFunctor unary_func           = UnaryFunctor{},
                                 std::int64_t work_items_per_group = -1) {
    reducer_singlepass<Float, BinaryFunctor, UnaryFunctor, layout::row_major> kernel(q,
                                                                                     binary_func,
                                                                                     unary_func);
    return (work_items_per_group <= 0)
               ? kernel(input, output, vector_size, n_vectors)
               : kernel(input, output, vector_size, n_vectors, work_items_per_group);
}

template <typename Float,
          typename BinaryFunctor,
          typename UnaryFunctor = unary_functor<Float, unary_operation::identity>>
inline cl::sycl::event reduce_cm(cl::sycl::queue& q,
                                 const Float* input,
                                 Float* output,
                                 std::int64_t vector_size,
                                 std::int64_t n_vectors,
                                 BinaryFunctor binary_func         = BinaryFunctor{},
                                 UnaryFunctor unary_func           = UnaryFunctor{},
                                 std::int64_t work_items_per_group = -1) {
    reducer_singlepass<Float, BinaryFunctor, UnaryFunctor, layout::col_major> kernel(q,
                                                                                     binary_func,
                                                                                     unary_func);
    return (work_items_per_group <= 0)
               ? kernel(input, output, vector_size, n_vectors)
               : kernel(input, output, vector_size, n_vectors, work_items_per_group);
}

template <typename Float,
          typename BinaryFunctor,
          typename UnaryFunctor = unary_functor<Float, unary_operation::identity>>
inline cl::sycl::event reduce_cm(cl::sycl::queue& q,
                                 array<Float> input,
                                 array<Float> output,
                                 std::int64_t vector_size,
                                 std::int64_t n_vectors,
                                 BinaryFunctor binary_func         = BinaryFunctor{},
                                 UnaryFunctor unary_func           = UnaryFunctor{},
                                 std::int64_t work_items_per_group = -1) {
    reducer_singlepass<Float, BinaryFunctor, UnaryFunctor, layout::col_major> kernel(q,
                                                                                     binary_func,
                                                                                     unary_func);
    return (work_items_per_group <= 0)
               ? kernel(input, output, vector_size, n_vectors)
               : kernel(input, output, vector_size, n_vectors, work_items_per_group);
}

#endif

} // namespace oneapi::dal::backend::primitives
