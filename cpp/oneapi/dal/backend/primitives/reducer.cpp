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

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>

#ifdef ONEAPI_DAL_DATA_PARALLEL
    #include <CL/sycl.hpp>
#endif

#include "oneapi/dal/backend/primitives/reducer.hpp"
#include "oneapi/dal/data/array.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float>
struct unary_functor<Float, unary_operation::identity> {
    Float operator()(Float arg) const {
        return arg;
    }
};

template <typename Float>
struct unary_functor<Float, unary_operation::abs> {
    Float operator()(Float arg) const {
        return std::abs(arg);
    }
};

template <typename Float>
struct unary_functor<Float, unary_operation::square> {
    Float operator()(Float arg) const {
        return (arg * arg);
    }
};

//Direct instatiation
template struct unary_functor<float, unary_operation::identity>;
template struct unary_functor<double, unary_operation::identity>;
template struct unary_functor<float, unary_operation::square>;
template struct unary_functor<double, unary_operation::square>;
template struct unary_functor<float, unary_operation::abs>;
template struct unary_functor<double, unary_operation::abs>;

template <typename Float>
struct binary_functor<Float, binary_operation::sum> {
    constexpr static Float init_value = 0;

    Float operator()(Float a, Float b) const {
        return (a + b);
    }
};

template <typename Float>
struct binary_functor<Float, binary_operation::mul> {
    constexpr static Float init_value = 1;

    Float operator()(Float a, Float b) const {
        return (a * b);
    }
};

template <typename Float>
struct binary_functor<Float, binary_operation::max> {
    constexpr static Float init_value = std::numeric_limits<Float>::min();

    Float operator()(Float a, Float b) const {
        return std::max(a, b);
    }
};

template <typename Float>
struct binary_functor<Float, binary_operation::min> {
    constexpr static Float init_value = std::numeric_limits<Float>::max();

    Float operator()(Float a, Float b) const {
        return std::min(a, b);
    }
};

//Direct instatiation
template struct binary_functor<float, binary_operation::sum>;
template struct binary_functor<double, binary_operation::sum>;
template struct binary_functor<float, binary_operation::mul>;
template struct binary_functor<double, binary_operation::mul>;
template struct binary_functor<float, binary_operation::min>;
template struct binary_functor<double, binary_operation::min>;
template struct binary_functor<float, binary_operation::max>;
template struct binary_functor<double, binary_operation::max>;

#ifdef ONEAPI_DAL_DATA_PARALLEL

namespace impl {

template <unary_operation UnOp, binary_operation BinOp, typename Float, bool IsRowMajorLayout>
struct reducer_singlepass_kernel {
public:
    //Zero cost
    const binary_functor<Float, BinOp> binary;
    const unary_functor<Float, UnOp> unary;

public:
    reducer_singlepass_kernel(std::int64_t vector_size_,
                              size_t n_vectors_,
                              const Float* vectors_,
                              Float* reduces_,
                              local_acc_t partial_reduces_)
            : vector_size(vector_size_),
              n_vectors(n_vectors_),
              vectors(vectors_),
              reduces(reduces_),
              partial_reduces(partial_reduces_){};

    void operator()(cl::sycl::nd_item<2> idx) const {
        const std::uint32_t local_size = idx.get_local_range(0);

        std::uint32_t global_dim = 1;
        std::uint32_t local_dim  = n_vectors;

        if constexpr (IsRowMajorLayout) {
            global_dim = vector_size;
            local_dim  = 1;
        }

        const std::uint32_t item_id  = idx.get_local_id(0);
        const std::uint32_t group_id = idx.get_global_id(1);

        Float el                 = vectors[group_id * global_dim + item_id * local_dim];
        partial_reduces[item_id] = binary.init_value;

        for (std::uint32_t i = item_id; i < vector_size; i += local_size) {
            el                       = vectors[group_id * global_dim + i * local_dim];
            partial_reduces[item_id] = binary(partial_reduces[item_id], unary(el));
        }

        idx.barrier(cl::sycl::access::fence_space::local_space);

        for (std::uint32_t stride = local_size / 2; stride > 1; stride /= 2) {
            if (stride > item_id) {
                partial_reduces[item_id] =
                    binary(partial_reduces[item_id], partial_reduces[item_id + stride]);
            }

            idx.barrier(cl::sycl::access::fence_space::local_space);
        }

        if (item_id == 0) {
            reduces[group_id] = binary(partial_reduces[item_id], partial_reduces[item_id + 1]);
        }
    }

protected:
    typedef cl::sycl::
        accessor<Float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>
            local_acc_t;
    const std::uint32_t vector_size, n_vectors;
    const Float* vectors;
    Float* reduces;
    local_acc_t partial_reduces;
};

} // namespace impl

template <unary_operation UnOp, binary_operation BinOp, typename Float, bool IsRowMajorLayout>
reducer_singlepass<UnOp, BinOp, Float, IsRowMajorLayout>::reducer_singlepass(cl::sycl::queue& q)
        : _q(q) {}

template <unary_operation UnOp, binary_operation BinOp, typename Float, bool IsRowMajorLayout>
cl::sycl::event reducer_singlepass<UnOp, BinOp, Float, IsRowMajorLayout>::operator()(
    array<Float> input,
    array<Float> output,
    std::int64_t vector_size,
    std::int64_t n_vectors,
    std::int64_t work_items_per_group) {
    if (input.get_count() < (vector_size * n_vectors))
        throw std::exception();
    typedef impl::reducer_singlepass_kernel<UnOp, BinOp, Float, IsRowMajorLayout> kernel_t;
    const std::int64_t local_buff_size = 256;
    auto event                         = this->_q.submit([&](cl::sycl::handler& handler) {
        typedef cl::sycl::
            accessor<Float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>
                local_acc_t;
        auto partial_reduces  = local_acc_t(cl::sycl::range<1>{ local_buff_size }, handler);
        auto functor_instance = kernel_t{ .vector_size = static_cast<std::uint32_t>(vector_size),
                                          .n_vectors   = static_cast<std::uint32_t>(n_vectors),
                                          .vectors     = input.get_data(),
                                          .reduces     = output.get_mutable_data(),
                                          .partial_reduces = partial_reduces };
        const cl::sycl::range<2> local_range{ static_cast<size_t>(work_items_per_group), 1 };
        const cl::sycl::range<2> global_range{ static_cast<size_t>(work_items_per_group),
                                               static_cast<size_t>(n_vectors) };
        const cl::sycl::nd_range<2> call_range(local_range, global_range);
        handler.parallel_for(call_range, functor_instance);
    });
    return event;
}

template <unary_operation UnOp, binary_operation BinOp, typename Float, bool IsRowMajorLayout>
cl::sycl::event reducer_singlepass<UnOp, BinOp, Float, IsRowMajorLayout>::operator()(
    array<Float> input,
    array<Float> output,
    std::int64_t vector_size,
    std::int64_t n_vectors) {
    const std::int64_t work_items_per_group =
        this->_q.get_device().template get_info<cl::sycl::info::device::max_work_group_size>();
    return this->operator()(input, output, vector_size, n_vectors, work_items_per_group);
}

//Direct instantiation
template struct reducer_singlepass<unary_operation::abs, binary_operation::sum, float>;
template struct reducer_singlepass<unary_operation::abs, binary_operation::sum, double>;
template struct reducer_singlepass<unary_operation::square, binary_operation::sum, float>;
template struct reducer_singlepass<unary_operation::square, binary_operation::sum, double>;
template struct reducer_singlepass<unary_operation::abs, binary_operation::max, float>;
template struct reducer_singlepass<unary_operation::abs, binary_operation::max, double>;

#endif

} // namespace oneapi::dal::backend::primitives
