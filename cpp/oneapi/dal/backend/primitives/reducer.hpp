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
#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/functors.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEAPI_DAL_DATA_PARALLEL

namespace impl {

template <typename UnOp, typename BinOp, typename Float, bool VectorsAreRows>
struct reducer_singlepass_kernel;

}

template <typename Float,
          typename BinaryFunctor,
          typename UnaryFunctor = unary_functor<Float, unary_operator::identity>,
          data_layout Layout    = data_layout::row_major>
struct reducer_singlepass {
private:
    constexpr static inline bool vectors_are_rows = (Layout == data_layout::row_major);
    typedef impl::reducer_singlepass_kernel<UnOp, BinOp, Float, vectors_are_rows> kernel_t;
    
public:
    reducer_singlepass( cl::sycl::queue& q, 
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
    const Unary Functor unary;

private:
    cl::sycl::queue& q;
    const std::int64_t max_work_group_size;
};

#endif

} // namespace oneapi::dal::backend::primitives
