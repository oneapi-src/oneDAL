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

#include <tuple>
#include <daal/src/algorithms/kmeans/kmeans_init_kernel.h>

#include "oneapi/dal/algo/kmeans/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_integral.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/cluster_updater.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_fp.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/transfer.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include <iostream>
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::kmeans::backend {

using dal::backend::context_gpu;
using descriptor_t = detail::descriptor_base<task::clustering>;

namespace daal_kmeans_init = daal::algorithms::kmeans::init;
namespace interop = dal::backend::interop;
namespace pr = dal::backend::primitives;
namespace de = dal::detail;
namespace bk = dal::backend;

template <typename Float, daal::CpuType Cpu>
using daal_kmeans_init_plus_plus_dense_kernel_t =
    daal_kmeans_init::internal::KMeansInitKernel<daal_kmeans_init::plusPlusDense, Float, Cpu>;

template <typename Float>
static pr::ndarray<Float, 2> get_initial_centroids(const dal::backend::context_gpu& ctx,
                                                   const descriptor_t& params,
                                                   const train_input<task::clustering>& input) {
    auto& queue = ctx.get_queue();

    const auto data = input.get_data();

    const std::int64_t column_count = data.get_column_count();
    const std::int64_t cluster_count = params.get_cluster_count();

    daal::data_management::NumericTablePtr daal_initial_centroids;

    if (!input.get_initial_centroids().has_data()) {
        // We use CPU algorithm for initialization, so input data
        // may be copied to DAAL homogen table
        const auto daal_data = interop::copy_to_daal_homogen_table<Float>(data);
        daal_kmeans_init::Parameter par(dal::detail::integral_cast<std::size_t>(cluster_count));

        const std::size_t init_len_input = 1;
        daal::data_management::NumericTable* init_input[init_len_input] = { daal_data.get() };

        daal_initial_centroids =
            interop::allocate_daal_homogen_table<Float>(cluster_count, column_count);
        const std::size_t init_len_output = 1;
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
        daal::data_management::BlockDescriptor<Float> block;
        daal_initial_centroids->getBlockOfRows(0,
                                               cluster_count,
                                               daal::data_management::readOnly,
                                               block);
        Float* initial_centroids_ptr = block.getBlockPtr();
        auto arr_host_initial =
            pr::ndarray<Float, 2>::wrap(initial_centroids_ptr, { cluster_count, column_count });
        return arr_host_initial.to_device(queue);
    }
    auto initial_centroids_ptr = row_accessor<const Float>(input.get_initial_centroids())
                                     .pull(queue, { 0, -1 }, sycl::usm::alloc::device);
    return pr::ndarray<Float, 2>::wrap(initial_centroids_ptr, { cluster_count, column_count });
}

template <typename Float>
struct train_kernel_gpu<Float, method::lloyd_dense, task::clustering> {
    train_result<task::clustering> operator()(const dal::backend::context_gpu& ctx,
                                              const descriptor_t& params,
                                              const train_input<task::clustering>& input) const {
        auto& queue = ctx.get_queue();
        auto& comm = ctx.get_communicator();

        const auto data = input.get_data();
        const std::int64_t row_count = data.get_row_count();
        const std::int64_t column_count = data.get_column_count();
        const std::int64_t cluster_count = params.get_cluster_count();
        const std::int64_t max_iteration_count = params.get_max_iteration_count();
        const double accuracy_threshold = params.get_accuracy_threshold();
        dal::detail::check_mul_overflow(cluster_count, column_count);
        std::cout << "here" << std::endl;
        auto data_ptr =
            row_accessor<const Float>(data).pull(queue, { 0, -1 }, sycl::usm::alloc::device);
        auto arr_data = pr::ndarray<Float, 2>::wrap(data_ptr, { row_count, column_count });
        std::cout << "here pull end" << std::endl;

        // TODO: Use truly-distributed algorithm for computing initial centroids.
        // The current implementation of distributed algorithm initializes centroids
        // independently on each rank using the data available. This may result in
        // inconsistent results between single-rank and distributed runs. To fix
        // this issue the correct distributed implementation of K-Means++ should be
        // called underneath.
        auto arr_initial = get_initial_centroids<Float>(ctx, params, input);

        std::int64_t block_size_in_rows =
            std::min(row_count,
                     kernels_fp<Float>::get_block_size_in_rows(queue, column_count, cluster_count));

        dal::detail::check_mul_overflow(block_size_in_rows, cluster_count);
        std::int64_t part_count =
            kernels_fp<Float>::get_part_count_for_partial_centroids(queue,
                                                                    column_count,
                                                                    cluster_count);

        auto arr_centroid_squares =
            pr::ndarray<Float, 1>::empty(queue, cluster_count, sycl::usm::alloc::device);
        auto arr_data_squares =
            pr::ndarray<Float, 1>::empty(queue, row_count, sycl::usm::alloc::device);
        auto data_squares_event =
            kernels_fp<Float>::compute_squares(queue, arr_data, arr_data_squares);
        auto arr_distance_block =
            pr::ndarray<Float, 2>::empty(queue,
                                         { block_size_in_rows, cluster_count },
                                         sycl::usm::alloc::device);
        auto arr_closest_distances =
            pr::ndarray<Float, 2>::empty(queue, { row_count, 1 }, sycl::usm::alloc::device);
        auto arr_centroids = pr::ndarray<Float, 2>::empty(queue,
                                                          { cluster_count, column_count },
                                                          sycl::usm::alloc::device);
        auto arr_responses =
            pr::ndarray<std::int32_t, 2>::empty(queue, { row_count, 1 }, sycl::usm::alloc::device);
        auto arr_objective_function =
            pr::ndarray<Float, 1>::empty(queue, 1, sycl::usm::alloc::device);

        Float prev_objective_function = de::limits<Float>::max();
        std::int64_t iter;
        sycl::event centroids_event;

        auto updater = cluster_updater<Float>{ queue, comm }
                           .set_cluster_count(cluster_count)
                           .set_part_count(part_count)
                           .set_data(arr_data)
                           .set_data_squares(arr_data_squares);
        updater.allocate_buffers();

        for (iter = 0; iter < max_iteration_count; iter++) {
            auto centroid_squares_event =
                kernels_fp<Float>::compute_squares(queue,
                                                   iter == 0 ? arr_initial : arr_centroids,
                                                   arr_centroid_squares);
            updater.set_initial_centroids(iter == 0 ? arr_initial : arr_centroids);
            updater.set_centroid_squares(arr_centroid_squares);
            auto [objective_function, update_clusters_event] =
                updater.update(arr_centroids,
                               arr_distance_block,
                               arr_closest_distances,
                               arr_objective_function,
                               arr_responses,
                               { centroids_event, data_squares_event, centroid_squares_event });
            centroids_event = update_clusters_event;
            if (accuracy_threshold > 0 &&
                objective_function + accuracy_threshold > prev_objective_function) {
                iter++;
                break;
            }
            prev_objective_function = objective_function;
        }
        auto centroid_squares_event = kernels_fp<Float>::compute_squares(queue,
                                                                         arr_centroids,
                                                                         arr_centroid_squares,
                                                                         { centroids_event });
        auto assign_event = kernels_fp<Float>::assign_clusters(queue,
                                                               arr_data,
                                                               arr_centroids,
                                                               arr_data_squares,
                                                               arr_centroid_squares,
                                                               block_size_in_rows,
                                                               arr_responses,
                                                               arr_distance_block,
                                                               arr_closest_distances,
                                                               { centroid_squares_event });

        auto objective_event = kernels_fp<Float>::compute_objective_function( //
            queue,
            arr_closest_distances,
            arr_objective_function,
            { assign_event });

        Float final_objective_function =
            arr_objective_function.to_host(queue, { objective_event }).get_data()[0];

        {
            ONEDAL_PROFILER_TASK(allreduce_final_objective);
            comm.allreduce(final_objective_function).wait();
        }

        model<task::clustering> model;
        model.set_centroids(
            dal::homogen_table::wrap(arr_centroids.flatten(queue), cluster_count, column_count));
        return train_result<task::clustering>()
            .set_responses(dal::homogen_table::wrap(arr_responses.flatten(queue), row_count, 1))
            .set_iteration_count(iter)
            .set_objective_function_value(final_objective_function)
            .set_model(model);
    }
};

template struct train_kernel_gpu<float, method::lloyd_dense, task::clustering>;
template struct train_kernel_gpu<double, method::lloyd_dense, task::clustering>;

} // namespace oneapi::dal::kmeans::backend
