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

#define DAAL_SYCL_INTERFACE
#define DAAL_SYCL_INTERFACE_USM
#define DAAL_SYCL_INTERFACE_REVERSED_RANGE

#include <include/algorithms/kmeans/kmeans_init_types.h>
#include <src/algorithms/kmeans/oneapi/kmeans_init_dense_batch_kernel_ucapi.h>

#include "oneapi/dal/algo/kmeans_init/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/algo/kmeans_init/backend/to_daal_method.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::kmeans_init::backend {

using std::int64_t;
using dal::backend::context_gpu;

namespace daal_kmeans_init = daal::algorithms::kmeans::init;
namespace interop = dal::backend::interop;

template <typename Float, typename Method>
using daal_kmeans_init_kernel_t =
    daal_kmeans_init::internal::KMeansInitDenseBatchKernelUCAPI<to_daal_method<Method>::value,
                                                                Float>;

template <typename Float, typename Method, typename Task>
static compute_result<Task> call_daal_kernel(const context_gpu& ctx,
                                             const descriptor_base<Task>& params,
                                             const table& data) {
    if constexpr (std::is_same_v<Method, method::plus_plus_dense>)
        throw unimplemented_error("plus_plus_dense method is not implemented for GPU");
    if constexpr (std::is_same_v<Method, method::parallel_plus_dense>)
        throw unimplemented_error("parallel_plus_dense method is not implemented for GPU");

    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const int64_t column_count = data.get_column_count();
    const int64_t cluster_count = params.get_cluster_count();

    daal_kmeans_init::Parameter par(cluster_count);

    auto arr_data = row_accessor<const Float>{ data }.pull(queue);
    const auto daal_data = interop::convert_to_daal_sycl_homogen_table(queue,
                                                                       arr_data,
                                                                       data.get_row_count(),
                                                                       data.get_column_count());

    array<Float> arr_centroids = array<Float>::empty(queue, cluster_count * column_count);
    const auto daal_centroids = interop::convert_to_daal_sycl_homogen_table(queue,
                                                                            arr_centroids,
                                                                            cluster_count,
                                                                            column_count);

    const size_t len_daal_input = 1;
    daal::data_management::NumericTable* daal_input[len_daal_input] = { daal_data.get() };
    const size_t len_daal_output = 1;
    daal::data_management::NumericTable* daal_output[len_daal_output] = { daal_centroids.get() };

    interop::status_to_exception(daal_kmeans_init_kernel_t<Float, Method>().compute(len_daal_input,
                                                                                    daal_input,
                                                                                    len_daal_output,
                                                                                    daal_output,
                                                                                    &par,
                                                                                    *(par.engine)));

    return compute_result<Task>().set_centroids(
        dal::detail::homogen_table_builder{}
            .reset(arr_centroids, cluster_count, column_count)
            .build());
}

template <typename Float, typename Method, typename Task>
static compute_result<Task> compute(const context_gpu& ctx,
                                    const descriptor_base<Task>& desc,
                                    const compute_input<Task>& input) {
    return call_daal_kernel<Float, Method, Task>(ctx, desc, input.get_data());
}

template <typename Float, typename Method, typename Task>
compute_result<Task> compute_kernel_gpu<Float, Method, Task>::operator()(
    const context_gpu& ctx,
    const descriptor_base<Task>& desc,
    const compute_input<Task>& input) const {
    return compute<Float, Method, Task>(ctx, desc, input);
}

template struct compute_kernel_gpu<float, method::dense, task::init>;
template struct compute_kernel_gpu<double, method::dense, task::init>;
template struct compute_kernel_gpu<float, method::random_dense, task::init>;
template struct compute_kernel_gpu<double, method::random_dense, task::init>;
template struct compute_kernel_gpu<float, method::plus_plus_dense, task::init>;
template struct compute_kernel_gpu<double, method::plus_plus_dense, task::init>;
template struct compute_kernel_gpu<float, method::parallel_plus_dense, task::init>;
template struct compute_kernel_gpu<double, method::parallel_plus_dense, task::init>;

} // namespace oneapi::dal::kmeans_init::backend
