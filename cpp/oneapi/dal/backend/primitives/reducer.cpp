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
    constexpr static Float operator()(Float arg) {
        return arg;
    }
};

template <typename Float>
struct unary_functor<Float, unary_operation::abs> {
    constexpr static Float operator()(Float arg) {
        return std::abs(arg);
    }
};

template <typename Float>
struct unary_functor<Float, unary_operation::square> {
    constexpr static Float operator()(Float arg) {
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

    constexpr static Float operator()(Float a, Float b) {
        return (a + b);
    }
};

template <typename Float>
struct binary_functor<Float, binary_operation::mul> {
    constexpr static Float init_value = 1;

    constexpr static Float operator()(Float a, Float b) {
        return (a * b);
    }
};

template <typename Float>
struct binary_functor<Float, binary_operation::max> {
    constexpr static Float init_value = std::numeric_limits<Float>::min();

    constexpr static Float operator()(Float a, Float b) {
        return std::max(a, b);
    }
};

template <typename Float>
struct binary_functor<Float, binary_operation::min> {
    constexpr static Float init_value = std::numeric_limits<Float>::max();

    constexpr static Float operator()(Float a, Float b) {
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
    constexpr typename unary_functor<BinOp> binary;
    constexpr typename unary_functor<UnOp> unary;

public:
    void operator()(cl::sycl::nd_item<2> idx) const {
        const std::uint32_t local_size = idx.get_local_size(0);

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

        idx.barrier(cl::sycl::access::fence_space::local);

        for (std::uint32_t stride = local_size / 2; stride > 1; stride /= 2) {
            if (stride > item_id) {
                partial_reduces[item_id] =
                    binary(partial_reduces[item_id], partial_reduces[item_id + stride]);
            }

            idx.barrier(cl::sycl::access::fence_space::local);
        }

        if (item_id == 0) {
            reduces[group_id] = binary(partial_reduces[item_id], partial_reduces[item_id + 1]);
        }
    }

protected:
    typedef cl::sycl::
        accessor<Float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>
            local_acc_t;
    std::uint32_t vector_size, n_vectors;
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
    typedef impl::reducer_single_pass<UnOp, BinOp, Float, IsRowMajorLayout> kernel_t;
    const std::int64_t local_buff_size = 256;
    auto event                         = this->_q.submit([&](cl::sycl::handler& handler) {
        typedef cl::sycl::
            accessor<Float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>
                local_acc_t;
        auto partial_reduces  = local_acc_t(cl::sycl::range<1>{ local_buff_size }, handler);
        auto functor_instance = kernel_t{ .vector_size     = vector_size,
                                          .n_vectors       = n_vectors,
                                          .vectors         = input.get_data(),
                                          .reduces         = output.get_mutable_data(),
                                          .partial_reduces = partial_reduces };
        const cl::sycl::range<2> local_range{ static_cast<size_t>(work_items_per_group), 1 };
        const cl::sycl::range<2> global_range{ static_cast<size_t>(work_items_per_group),
                                               static_cast<size_t>(n_vectors) };
        const cl::sycl::nd_range<2> call_range(local_range, global_range);
        handler.parallel_for(call_range, functor_instance);
    });
    event.wait();
    return event;
}

template <unary_operation UnOp, binary_operation BinOp, typename Float, bool IsRowMajorLayout>
cl::sycl::event reducer_singlepass<UnOp, BinOp, Float, IsRowMajorLayout>::operator()(
    array<Float> input,
    array<Float> output,
    std::int64_t vector_size,
    std::int64_t n_vectors) {
    const std::int64_t work_items_per_group =
        this->_q.template get_info<cl::sycl::info::device::max_work_group_size>();
    return this->operator()(input, output, vector_size, vector_size, n_vectors);
}

template <unary_operation UnOp, binary_operation BinOp, typename Float, bool IsRowMajorLayout>
typename array<Float> reducer_singlepass<UnOp, BinOp, Float, IsRowMajorLayout>::operator()(
    array<const Float> input,
    std::int64_t vector_size,
    std::int64_t n_vectors) {
    auto output = array<Float>::zeros(this->_q, m);
    this->operator()(input, output, m, n);
    return output;
}

//Direct instantiation
template struct l1_reducer_singlepass<float>;
template struct l1_reducer_singlepass<double>;
template struct l2_reducer_singlepass<float>;
template struct l2_reducer_singlepass<double>;
template struct linf_reducer_singlepass<float>;
template struct linf_reducer_singlepass<double>;

#endif

} // namespace oneapi::dal::backend::primitives
