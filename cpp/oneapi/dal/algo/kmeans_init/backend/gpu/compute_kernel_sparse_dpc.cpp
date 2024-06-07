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

#include <daal/src/algorithms/kmeans/kmeans_init_kernel.h>
#include "oneapi/dal/algo/kmeans_init/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/algo/kmeans_init/backend/gpu/compute_kernel_distr.hpp"
#include "oneapi/dal/algo/kmeans_init/backend/gpu/compute_kernels_impl.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/table/csr_accessor.hpp"
#include "oneapi/dal/algo/kmeans_init/backend/to_daal_method.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::kmeans_init::backend {

using dal::backend::context_gpu;

namespace interop = dal::backend::interop;
namespace pr = dal::backend::primitives;
namespace ki = oneapi::dal::kmeans_init;

template <typename Float, daal::CpuType Cpu, typename Method>
using daal_kmeans_init_kernel_t =
    daal_kmeans_init::internal::KMeansInitKernel<to_daal_method<Method>::value, Float, Cpu>;

using task_t = task::init;
using input_t = compute_input<task_t>;
using result_t = compute_result<task_t>;
using descriptor_t = ki::detail::descriptor_base<task_t>;

template <typename Float, typename Method>
static result_t call_daal_kernel(const context_gpu& ctx,
                                 const descriptor_t& desc,
                                 const table& data) {
    auto& queue = ctx.get_queue();
    const std::int64_t column_count = data.get_column_count();
    const std::int64_t row_count = data.get_row_count();
    const std::int64_t cluster_count = desc.get_cluster_count();

    auto [csr_data, column_indices, row_offsets] =
        csr_accessor<const Float>(static_cast<const csr_table&>(data))
            .pull(queue, { 0, -1 }, sparse_indexing::one_based);
 
    auto csr_data_host =
        pr::ndarray<Float, 1>::wrap(csr_data.get_data(), csr_data.get_count()).to_host(queue);
    auto column_indices_host =
        pr::ndarray<std::int64_t, 1>::wrap(column_indices.get_data(), column_indices.get_count())
            .to_host(queue);
    auto row_offsets_host =
        pr::ndarray<std::int64_t, 1>::wrap(row_offsets.get_data(), row_offsets.get_count())
            .to_host(queue);
 
    table data_host = csr_table::wrap(queue,
                                     csr_data_host.get_data(),
                                     column_indices_host.get_data(),
                                     row_offsets_host.get_data(),
                                     row_count,
                                     column_count,
                                     sparse_indexing::one_based);


    //number of trials to pick each centroid from, 2 + int(ln(cluster_count)) works better than vanilla kmeans++
    //https://github.com/scikit-learn/scikit-learn/blob/a63b021310ba13ea39ad3555f550d8aeec3002c5/sklearn/cluster/_kmeans.py#L108
    std::int64_t trial_count = desc.get_local_trials_count();
    if (trial_count == -1) {
        const auto additional = std::log(cluster_count);
        trial_count = 2 + std::int64_t(additional);
    }

    daal_kmeans_init::Parameter par(dal::detail::integral_cast<std::size_t>(cluster_count),
                                    0,
                                    dal::detail::integral_cast<std::size_t>(desc.get_seed()));
    par.nTrials = trial_count;

    const auto daal_data = interop::convert_to_daal_table<Float>(data_host);
    const std::size_t len_input = 1;
    daal::data_management::NumericTable* input[len_input] = { daal_data.get() };

    dal::detail::check_mul_overflow(cluster_count, column_count);
    array<Float> arr_centroids = array<Float>::empty(cluster_count * column_count);
    const auto daal_centroids =
        interop::convert_to_daal_homogen_table(arr_centroids, cluster_count, column_count);
    const std::size_t len_output = 1;
    daal::data_management::NumericTable* output[len_output] = { daal_centroids.get() };

    const dal::backend::context_cpu cpu_ctx;
    interop::status_to_exception(dal::backend::dispatch_by_cpu(cpu_ctx, [&](auto cpu) {
        return daal_kmeans_init_kernel_t<
                   Float,
                   oneapi::dal::backend::interop::to_daal_cpu_type<decltype(cpu)>::value,
                   Method>()
            .compute(len_input, input, len_output, output, &par, *(par.engine));
    }));
    
    auto element_count = cluster_count * column_count;
    auto arr_centroids_device = dal::array<float>::empty(queue, element_count, sycl::usm::alloc::device);
    auto* const arr_centroids_ptr = arr_centroids_device.get_mutable_data();
    auto copy_to_device_event = queue.submit([&](sycl::handler& cgh) {
        cgh.memcpy(arr_centroids_ptr, arr_centroids.get_data(), element_count * sizeof(float));
    });
    
    return compute_result<task_t>().set_centroids(
        dal::detail::homogen_table_builder{}
            .reset(arr_centroids_device, cluster_count, column_count)
            .build());
}

template <typename Float, typename Method>
static result_t compute(const context_gpu& ctx, 
                        const descriptor_t& desc, 
                        const input_t& input) {
    return call_daal_kernel<Float, Method>(ctx, desc, input.get_data());
}

template <typename Float, typename Method>
struct compute_kernel_gpu<Float, Method, task::init> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float, Method>(ctx, desc, input);
    }
};

template struct compute_kernel_gpu<float, method::random_csr, task::init>;
template struct compute_kernel_gpu<double, method::random_csr, task::init>;
template struct compute_kernel_gpu<float, method::plus_plus_csr, task::init>;
template struct compute_kernel_gpu<double, method::plus_plus_csr, task::init>;
template struct compute_kernel_gpu<float, method::parallel_plus_csr, task::init>;
template struct compute_kernel_gpu<double, method::parallel_plus_csr, task::init>;

} // namespace oneapi::dal::kmeans_init::backend
