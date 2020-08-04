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

#include <cmath>
#include <limits>
#include <algorithm>

#include "oneapi/dal/backend/primirives/reducer.hpp"

namespace oneapi::dal::backend::primitives {

template<typename Float>
constexpr Float unary_functor<Float, unary_operation::identity>::operator() (Float arg) {
    return arg;
}

template<typename Float>
constexpr Float unary_functor<Float, unary_operation::abs>::operator() (Float arg) {
    return std::abs(arg);
}

template<typename Float>
constexpr Float unary_functor<Float, unary_operation::square>::operator() (Float arg) {
    return (arg * arg);
}

//Direct instatiation
template struct unary_functor<float, unary_operation::identity>;
template struct unary_functor<double, unary_operation::identity>;
template struct unary_functor<float, unary_operation::square>;
template struct unary_functor<double, unary_operation::square>;
template struct unary_functor<float, unary_operation::abs>;
template struct unary_functor<double, unary_operation::abs>;

template<typename Float>
constexpr Float binary_functor<Float, binary_operation::sum>::init_value = 0;

template<typename Float>
constexpr Float binary_functor<Float, binary_operation::mul>::init_value = 1;

template<typename Float>
constexpr Float binary_functor<Float, binary_operation::min>::init_value = std::numeric_limits<Float>::max();

template<typename Float>
constexpr Float binary_functor<Float, binary_operation::max>::init_value = std::numeric_limits<Float>::min();

template<typename Float, binary_operation Op>
constexpr Float binary_functor<Float, Op>::init_value = 0;

template<typename Float>
constexpr Float binary_functor<Float, binary_operation::sum>::operator() (Float a) {
    return operator()(init_value, a);
}

template<typename Float>
constexpr Float binary_functor<Float, binary_operation::mul>::operator() (Float a, Float b) {
    return (a * b);
}

template<typename Float>
constexpr Float binary_functor<Float, binary_operation::min>::operator() (Float a, Float b) {
    return std::min(a, b);
}

template<typename Float>
constexpr Float binary_functor<Float, binary_operation::max>::operator() (Float a, Float b) {
    return std::max(a, b);
}

//Direct instatiation
template struct binary_functor<float, binary::sum>;
template struct binary_functor<double, binary::sum>;
template struct binary_functor<float, binary::mul>;
template struct binary_functor<double, binary::mul>;
template struct binary_functor<float, binary::min>;
template struct binary_functor<double, binary::max>;

template<typename Float, unary_operation UnOp, binary_operation BinOp>
constexpr Float composed_functor<Float, UnOp, BinOp>::operator() (Float a, Float b) {
    Float unary_res = unary_functor<Float, UnOp>::operator()(b);
    return binary_functor<Float, BinOp>::operator()(a, std::move(unary_res)); 
}

template<typename Float, unary_operation UnOp, binary_operation BinOp>
constexpr Float composed_functor<Float, UnOp, BinOp>::operator() (Float a) {
    Float unary_res = unary_functor<Float, UnOp>::operator()(b);
    return binary_functor<Float, BinOp>::operator()(std::move(unary_res)); 
}

//Direct instantiation
template struct l1_functor<float>;
template struct l1_functor<double>;
template struct l2_functor<float>;
template struct l2_functor<double>;
template struct linf_functor<float>;
template struct linf_functor<double>;

namespace impl {

template<unary_operation UnOp, binary_operation BinOp, typename Float, bool RowMajorLayout = true>
struct reducer_singlepass_kernel
{
    //Zero cost
    constexpr unary_functor<BinOp> binary;
    constexpr unary_functor<UnOp> unary;
        
    void operator() (cl::sycl::nd_item<2> idx) const {
        const std::uint32_t local_size = idx.get_local_size(0);

        std::uint32_t global_dim = 1;
        std::uint32_t local_dim  = n_vectors;

        if constexpr (RowMajorLayout) {
            global_dim = vector_size;
            local_dim  = 1;
        }

        const std::uint32_t item_id  = idx.get_local_id(0);
        const std::uint32_t group_id = idx.get_global_id(1);

        Float el                = vectors[group_id * global_dim + item_id * local_dim];
        partial_reduces[item_id] = binary.init_value;

        for(std::uint32_t i = item_id; i < vector_size; i += local_size) {
            el                       = vectors[group_id * global_dim + i * local_dim];
            partial_reduces[item_id] = binary(partialReduces[item_id], unary(el));
        }

        idx.barrier(cl::sycl::access::fence_space::local);

        for(std::uint32_t stride = local_size / 2; stride > 1; stride /= 2) {
            if(stride > item_id) {
                partial_reduces[item_id] = binary(partial_reduces[item_id], partial_reduces[item_id + stride]);
            }

            idx.barrier(cl::sycl::access::fence_space::local);
        }

        if(item_id == 0) {
            reduces[group_id] = binary(partial_reduces[item_id], partial_reduces[item_id + 1]);
        }
    }
    protected:
        std::uint32_t n_vectors, vector_size;
        cl::sycl::accessor<algorithmFPType, 1, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer> vectors;
        cl::sycl::accessor<algorithmFPType, 1, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer> reduces;
        cl::sycl::accessor<algorithmFPType, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> partialReduces;
    };

} // oneapi::dal::backend::primitives