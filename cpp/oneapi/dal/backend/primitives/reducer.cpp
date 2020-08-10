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

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/reducer.hpp"
#include "oneapi/dal/data/array.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEAPI_DAL_DATA_PARALLEL

namespace impl {

template <typename UnaryFunctor, typename BinaryFunctor, typename Float, bool VectorsAreRows>
struct reducer_singlepass_kernel {
public:
    const BinaryFunctor binary;
    const UnaryFunctor unary;

private:
    typedef cl::sycl::
        accessor<Float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>
            local_acc_t;

public:
    reducer_singlepass_kernel(BinaryFunctor binary_,
                              UnaryFunctor unary_,
                              std::int64_t vector_size_,
                              size_t n_vectors_,
                              const Float* vectors_,
                              Float* reduces_,
                              local_acc_t partial_reduces_)
            : binary(binary_),
              unary(unary_),
              vector_size(vector_size_),
              n_vectors(n_vectors_),
              vectors(vectors_),
              reduces(reduces_),
              partial_reduces(partial_reduces_){};

    void operator()(cl::sycl::nd_item<2> idx) {
        const std::uint32_t local_size = idx.get_local_range(0);

        const std::uint32_t global_dim = VectorsAreRows ? vector_size : 1;
        const std::uint32_t local_dim  = VectorsAreRows ? 1 : n_vectors;

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

private:
    const std::uint32_t vector_size, n_vectors;
    const Float* const vectors;
    Float* const reduces;
    local_acc_t partial_reduces;
};

} // namespace impl

template <typename Float, typename BinaryFunctor, typename UnaryFunctor, layout Layout>
reducer_singlepass<Float, BinaryFunctor, UnaryFunctor, Layout>::reducer_singlepass(
    cl::sycl::queue& queue,
    BinaryFunctor binary_func,
    UnaryFunctor unary_func)
        : binary(binary_func),
          unary(unary_func),
          q(queue),
          max_work_group_size(std::min<std::int64_t>(
              queue.get_device().template get_info<cl::sycl::info::device::max_work_group_size>(),
              256)) {}

template <typename Float, typename BinaryFunctor, typename UnaryFunctor, layout Layout>
cl::sycl::event reducer_singlepass<Float, BinaryFunctor, UnaryFunctor, Layout>::operator()(
    const Float* input,
    Float* output,
    std::int64_t vector_size,
    std::int64_t n_vectors,
    std::int64_t work_items_per_group) {
    const std::int64_t local_buff_size = std::min(this->max_work_group_size, work_items_per_group);
    auto event                         = this->q.submit([&](cl::sycl::handler& handler) {
        typedef cl::sycl::
            accessor<Float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>
                local_acc_t;
        auto partial_reduces = local_acc_t(cl::sycl::range<1>{ local_buff_size }, handler);
        auto functor_instance =
            kernel_t{ /*.binary          =*/binary,
                      /*.unary           =*/unary,
                      /*.vector_size     =*/static_cast<std::uint32_t>(vector_size),
                      /*.n_vectors       =*/static_cast<std::uint32_t>(n_vectors),
                      /*.vectors         =*/input,
                      /*.reduces         =*/output,
                      /*.partial_reduces =*/partial_reduces };
        const cl::sycl::range<2> local_range{ static_cast<size_t>(work_items_per_group), 1 };
        const cl::sycl::range<2> global_range{ static_cast<size_t>(work_items_per_group),
                                               static_cast<size_t>(n_vectors) };
        const cl::sycl::nd_range<2> call_range(global_range, local_range);
        handler.parallel_for<kernel_t>(call_range, functor_instance);
    });
    return event;
}

template <typename Float, typename BinaryFunctor, typename UnaryFunctor, layout Layout>
cl::sycl::event reducer_singlepass<Float, BinaryFunctor, UnaryFunctor, Layout>::operator()(
    array<Float> input,
    array<Float> output,
    std::int64_t vector_size,
    std::int64_t n_vectors,
    std::int64_t work_items_per_group) {
    if (work_items_per_group > this->max_work_group_size)
        throw std::exception();
    if (input.get_count() < (vector_size * n_vectors))
        throw std::exception();
    if (output.get_count() < (vectors_are_rows ? n_vectors : vector_size))
        throw std::exception();
    return this->operator()(input.get_data(),
                            output.get_mutable_data(),
                            vector_size,
                            n_vectors,
                            work_items_per_group);
}

template <typename Float, typename BinaryFunctor, typename UnaryFunctor, layout Layout>
cl::sycl::event reducer_singlepass<Float, BinaryFunctor, UnaryFunctor, Layout>::operator()(
    const Float* input,
    Float* output,
    std::int64_t vector_size,
    std::int64_t n_vectors) {
    return this->operator()(input, output, vector_size, n_vectors, this->max_work_group_size);
}

template <typename Float, typename BinaryFunctor, typename UnaryFunctor, layout Layout>
cl::sycl::event reducer_singlepass<Float, BinaryFunctor, UnaryFunctor, Layout>::operator()(
    array<Float> input,
    array<Float> output,
    std::int64_t vector_size,
    std::int64_t n_vectors) {
    return this->operator()(input, output, vector_size, n_vectors, this->max_work_group_size);
}

//Direct instantiation
template struct reducer_singlepass<float,
                                   binary_functor<float, binary_operation::sum>,
                                   unary_functor<float, unary_operation::identity>,
                                   layout::row_major>;
template struct reducer_singlepass<double,
                                   binary_functor<double, binary_operation::sum>,
                                   unary_functor<double, unary_operation::identity>,
                                   layout::row_major>;
template struct reducer_singlepass<float,
                                   binary_functor<float, binary_operation::sum>,
                                   unary_functor<float, unary_operation::identity>,
                                   layout::col_major>;
template struct reducer_singlepass<double,
                                   binary_functor<double, binary_operation::sum>,
                                   unary_functor<double, unary_operation::identity>,
                                   layout::col_major>;
template struct reducer_singlepass<float,
                                   binary_functor<float, binary_operation::sum>,
                                   unary_functor<float, unary_operation::square>,
                                   layout::row_major>;
template struct reducer_singlepass<double,
                                   binary_functor<double, binary_operation::sum>,
                                   unary_functor<double, unary_operation::square>,
                                   layout::row_major>;
template struct reducer_singlepass<float,
                                   binary_functor<float, binary_operation::sum>,
                                   unary_functor<float, unary_operation::square>,
                                   layout::col_major>;
template struct reducer_singlepass<double,
                                   binary_functor<double, binary_operation::sum>,
                                   unary_functor<double, unary_operation::square>,
                                   layout::col_major>;

#endif

} // namespace oneapi::dal::backend::primitives
