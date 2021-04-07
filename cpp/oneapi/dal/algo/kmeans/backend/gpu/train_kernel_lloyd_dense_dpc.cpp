/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <daal/src/algorithms/kmeans/kmeans_init_kernel.h>
#include <src/algorithms/kmeans/oneapi/kmeans_dense_lloyd_batch_kernel_ucapi.h>

#include "oneapi/dal/algo/kmeans/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/clustering_by_blocks.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/backend/transfer.hpp"

namespace oneapi::dal::kmeans::backend {

using std::int64_t;
using dal::backend::context_gpu;
using descriptor_t = detail::descriptor_base<task::clustering>;

namespace daal_kmeans = daal::algorithms::kmeans;
namespace daal_kmeans_init = daal::algorithms::kmeans::init;
namespace interop = dal::backend::interop;

template <typename Float>
using daal_kmeans_lloyd_dense_ucapi_kernel_t =
    daal_kmeans::internal::KMeansDenseLloydBatchKernelUCAPI<Float>;

template <typename Float, daal::CpuType Cpu>
using daal_kmeans_init_plus_plus_dense_kernel_t =
    daal_kmeans_init::internal::KMeansInitKernel<daal_kmeans_init::plusPlusDense, Float, Cpu>;

template <typename Float>
static NumericTablePtr get_initial_centroids(const dal::backend::context_gpu& ctx,
                                             const descriptor_t& params,
                                             const train_input<task::clustering>& input) {
    auto& queue = ctx.get_queue();
    interop::execution_context_guard guard(queue);

    const auto data = input.get_data();

    const int64_t column_count = data.get_column_count();
    const int64_t cluster_count = params.get_cluster_count();

    daal::data_management::NumericTablePtr daal_initial_centroids;

    if (!input.get_initial_centroids().has_data()) {
        // We use CPU algorithm for initialization, so input data
        // may be copied to DAAL homogen table
        const auto daal_data = interop::copy_to_daal_homogen_table<Float>(data);
        daal_kmeans_init::Parameter par(dal::detail::integral_cast<std::size_t>(cluster_count));

        const size_t init_len_input = 1;
        daal::data_management::NumericTable* init_input[init_len_input] = { daal_data.get() };

        daal_initial_centroids =
            interop::allocate_daal_homogen_table<Float>(cluster_count, column_count);
        const size_t init_len_output = 1;
        daal::data_management::NumericTable* init_output[init_len_output] = {
            daal_initial_centroids.get()
        };

        const dal::backend::context_cpu cpu_ctx;
        interop::status_to_exception(
            interop::call_daal_kernel<Float, daal_kmeans_init_plus_plus_dense_kernel_t>(
                cpu_ctx,
                init_len_input,
                init_input,
                init_len_output,
                init_output,
                &par,
                *(par.engine)));
    }
    else {
        daal_initial_centroids =
            interop::convert_to_daal_table(queue, input.get_initial_centroids());
    }
    return daal_initial_centroids;
}

template <typename Float>
struct train_kernel_gpu<Float, method::lloyd_dense, task::clustering> {
/*    train_result<task::clustering> operator()(const dal::backend::context_gpu& ctx,
                                              const descriptor_t& params,
                                              const train_input<task::clustering>& input) const {
        const auto data = input.get_data();

        const int64_t row_count = data.get_row_count();
        const int64_t column_count = data.get_column_count();

        const int64_t cluster_count = params.get_cluster_count();
        const int64_t max_iteration_count = params.get_max_iteration_count();
        const double accuracy_threshold = params.get_accuracy_threshold();

        auto daal_initial_centroids = get_initial_centroids<Float>(ctx, params, input);

        auto& queue = ctx.get_queue();
        interop::execution_context_guard guard(queue);

        daal_kmeans::Parameter par(dal::detail::integral_cast<std::size_t>(cluster_count),
                                   dal::detail::integral_cast<std::size_t>(max_iteration_count));
        par.accuracyThreshold = accuracy_threshold;

        const auto daal_data = interop::convert_to_daal_table(queue, data);

        dal::detail::check_mul_overflow(cluster_count, column_count);
        array<Float> arr_centroids =
            array<Float>::empty(queue, cluster_count * column_count, sycl::usm::alloc::device);
        array<Float> arr_obj_func = array<Float>::empty(queue, 1, sycl::usm::alloc::device);
        array<int> arr_labels = array<int>::empty(queue, row_count, sycl::usm::alloc::device);
        array<int> arr_iter_count = array<int>::empty(queue, 1, sycl::usm::alloc::device);

        const auto daal_labels = interop::convert_to_daal_table(queue, arr_labels, row_count, 1);
        const auto daal_objective_function_value =
            interop::convert_to_daal_table(queue, arr_obj_func, 1, 1);
        const auto daal_iteration_count =
            interop::convert_to_daal_table(queue, arr_iter_count, 1, 1);

        const auto daal_centroids =
            interop::convert_to_daal_table(queue, arr_centroids, cluster_count, column_count);

        daal::data_management::NumericTable* daal_input[2] = { daal_data.get(),
                                                               daal_initial_centroids.get() };

        daal::data_management::NumericTable* daal_output[4] = { daal_centroids.get(),
                                                                daal_labels.get(),
                                                                daal_objective_function_value.get(),
                                                                daal_iteration_count.get() };
        interop::status_to_exception(
            daal_kmeans_lloyd_dense_ucapi_kernel_t<Float>().compute(daal_input, daal_output, &par));

        auto [arr_iter_count_host, arr_iter_count_event] = dal::backend::to_host(arr_iter_count);
        auto [arr_obj_func_host, arr_obj_func_event] = dal::backend::to_host(arr_obj_func);
        sycl::event::wait_and_throw({ arr_iter_count_event, arr_obj_func_event });

        return train_result<task::clustering>()
            .set_labels(
                dal::detail::homogen_table_builder{}.reset(arr_labels, row_count, 1).build())
            .set_iteration_count(static_cast<std::int64_t>(arr_iter_count_host[0]))
            .set_objective_function_value(static_cast<double>(arr_obj_func_host[0]))
            .set_model(model<task::clustering>().set_centroids(
                dal::detail::homogen_table_builder{}
                    .reset(arr_centroids, cluster_count, column_count)
                    .build()));
    }*/
    train_result<task::clustering> operator()(const dal::backend::context_gpu& ctx,
                                              const descriptor_t& params,
                                              const train_input<task::clustering>& input) const {
        const auto data = input.get_data();

        const int64_t row_count = data.get_row_count();
        const int64_t column_count = data.get_column_count();

        const int64_t cluster_count = params.get_cluster_count();
        const int64_t max_iteration_count = params.get_max_iteration_count();
        const double accuracy_threshold = params.get_accuracy_threshold();

        auto initial_centroids = get_initial_centroids<Float>(ctx, params, input);

        auto& queue = ctx.get_queue();
        interop::execution_context_guard guard(queue);

        dal::detail::check_mul_overflow(cluster_count, column_count);

        ndarray<Float> arr_centroids =
            ndarray<Float,2>::empty(queue, {cluster_count, column_count}, sycl::usm::alloc::device);
        ndarray<std::int32_t> arr_labels = ndarray<std::int32_t, 1>::empty(queue, row_count, sycl::usm::alloc::device);
        ndarray<Float> arr_distances = ndarray<Float, 1>::empty(queue, row_count, sycl::usm::alloc::device);


        kmeans_core<Float> estimator(queue, row_count, column_count, desc);
        Float prev_objective_function = detail::limits<Float>::max();
        std::int64_t iter;
        sycl::event centroids_event;
        for(iter = 0; iter < max_iter_num; iter++)
        {
            auto [assign_event, count_event, objective_function_event] = 
                estimator.make_clusters(data, iter == 0 ? initial_centroids : arr_centorids, arr_labels, arr_distances, {centroids_event});
            centroids_event = estimator.reduce_centroids(data, arr_labels, arr_centroids, {assign_event});
            count_event.wait_and_throw();
            if(estimator.get_num_empty_clusters() > 0) {
                centroids_event = estimator.findCandidates(arr_labels, arr_distances, {centroids_event});
            }
            objective_function_event.wait_and_throw();
            if(estimator.get_objective_function() + accuracy_threshold > prev_objective_function) {
                iter++;
                break;
            }
            prev_objective_function = estimator.get_objective_function();
        }
        auto [assign_event, count_event, objective_function_event] = 
            estimator.make_clusters(data, arr_centorids, arr_labels, arr_distances, {centroids_event});
        {assign_event, count_event, objective_function_event}.wait_and_throw();
        return train_result<task::clustering>()
            .set_labels(
                dal::detail::homogen_table_builder{}.reset(arr_labels, row_count, 1).build())
            .set_iteration_count(iter)
            .set_objective_function_value(estimator.get_objective_function())
            .set_model(model<task::clustering>().set_centroids(
                dal::detail::homogen_table_builder{}
                    .reset(arr_centroids, cluster_count, column_count)
                    .build()));
    }    
};

template struct train_kernel_gpu<float, method::lloyd_dense, task::clustering>;
template struct train_kernel_gpu<double, method::lloyd_dense, task::clustering>;

} // namespace oneapi::dal::kmeans::backend
