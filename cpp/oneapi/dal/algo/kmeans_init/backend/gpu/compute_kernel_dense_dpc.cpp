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

#include "oneapi/dal/algo/kmeans_init/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/algo/kmeans_init/backend/gpu/compute_kernel_distr.hpp"
#include "oneapi/dal/algo/kmeans_init/backend/gpu/compute_kernels_impl.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::kmeans_init::backend {

using dal::backend::context_gpu;

namespace interop = dal::backend::interop;
namespace pr = dal::backend::primitives;
namespace ki = oneapi::dal::kmeans_init;

using task_t = task::init;
using input_t = compute_input<task_t>;
using result_t = compute_result<task_t>;
using descriptor_t = ki::detail::descriptor_base<task_t>;

template <typename Float, typename Method>
static result_t call_daal_kernel(const context_gpu& ctx,
                                 const descriptor_t& params,
                                 const table& data) {
    auto& queue = ctx.get_queue();

    const std::int64_t column_count = data.get_column_count();
    const std::int64_t cluster_count = params.get_cluster_count();

    auto arr_data = pr::table2ndarray<Float>(queue, data, sycl::usm::alloc::device);

    dal::detail::check_mul_overflow(cluster_count, column_count);
    auto arr_centroids = pr::ndarray<Float, 2>::empty(queue,
                                                      { cluster_count, column_count },
                                                      sycl::usm::alloc::device);

    kmeans_init_kernel<Float, Method>::compute_initial_centroids(ctx, arr_data, arr_centroids)
        .wait_and_throw();
    return result_t().set_centroids(
        dal::homogen_table::wrap(arr_centroids.flatten(queue), cluster_count, column_count));
}

template <typename Float, typename Method>
static result_t compute(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    constexpr bool is_random_dense = std::is_same_v<Method, method::random_dense>;
    constexpr bool is_plus_plus_dense = std::is_same_v<Method, method::plus_plus_dense>;
    using distr_t = compute_kernel_distr<Float, Method, task_t>;
    if constexpr (is_random_dense || is_plus_plus_dense) {
        return distr_t{}(ctx, desc, input);
    }
    else {
        return call_daal_kernel<Float, Method>(ctx, desc, input.get_data());
    }
}

template <typename Float, typename Method>
struct compute_kernel_gpu<Float, Method, task::init> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float, Method>(ctx, desc, input);
    }
};

template struct compute_kernel_gpu<float, method::dense, task::init>;
template struct compute_kernel_gpu<double, method::dense, task::init>;
template struct compute_kernel_gpu<float, method::random_dense, task::init>;
template struct compute_kernel_gpu<double, method::random_dense, task::init>;
template struct compute_kernel_gpu<float, method::plus_plus_dense, task::init>;
template struct compute_kernel_gpu<double, method::plus_plus_dense, task::init>;
template struct compute_kernel_gpu<float, method::parallel_plus_dense, task::init>;
template struct compute_kernel_gpu<double, method::parallel_plus_dense, task::init>;

} // namespace oneapi::dal::kmeans_init::backend
