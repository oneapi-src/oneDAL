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

#include "oneapi/dal/algo/kmeans/backend/gpu/infer_kernel.hpp"
#include "oneapi/dal/backend/transfer.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_integral.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kernels_fp.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::kmeans::backend {

using dal::backend::context_gpu;
using descriptor_t = detail::descriptor_base<task::clustering>;

namespace pr = dal::backend::primitives;

template <typename Float>
struct infer_kernel_gpu<Float, method::lloyd_dense, task::clustering> {
    infer_result<task::clustering> operator()(const dal::backend::context_gpu& ctx,
                                              const descriptor_t& params,
                                              const infer_input<task::clustering>& input) const {
        auto& queue = ctx.get_queue();
        ONEDAL_PROFILER_TASK(kmeans.infer_kernel, queue);
        const auto data = input.get_data();
        const std::int64_t row_count = data.get_row_count();
        const std::int64_t column_count = data.get_column_count();
        const std::int64_t cluster_count = params.get_cluster_count();

        auto arr_data = pr::table2ndarray<Float>(queue, data, sycl::usm::alloc::device);
        auto arr_centroids = pr::table2ndarray<Float>(queue,
                                                      input.get_model().get_centroids(),
                                                      sycl::usm::alloc::device);

        std::int64_t block_size_in_rows =
            std::min(row_count,
                     kernels_fp<Float>::get_block_size_in_rows(queue, column_count, cluster_count));
        dal::detail::check_mul_overflow(block_size_in_rows, cluster_count);
        auto arr_distance_block =
            pr::ndarray<Float, 2>::empty(queue,
                                         { block_size_in_rows, cluster_count },
                                         sycl::usm::alloc::device);
        auto arr_centroid_squares =
            pr::ndarray<Float, 1>::empty(queue, cluster_count, sycl::usm::alloc::device);
        auto arr_data_squares =
            pr::ndarray<Float, 1>::empty(queue, row_count, sycl::usm::alloc::device);
        auto data_squares_event =
            kernels_fp<Float>::compute_squares(queue, arr_data, arr_data_squares);
        auto centroid_squares_event =
            kernels_fp<Float>::compute_squares(queue, arr_centroids, arr_centroid_squares);
        auto arr_closest_distances =
            pr::ndarray<Float, 2>::empty(queue, { row_count, 1 }, sycl::usm::alloc::device);
        auto arr_responses =
            pr::ndarray<std::int32_t, 2>::empty(queue, { row_count, 1 }, sycl::usm::alloc::device);
        auto arr_objective_function =
            pr::ndarray<Float, 1>::empty(queue, 1, sycl::usm::alloc::device);

        auto assign_event =
            kernels_fp<Float>::assign_clusters(queue,
                                               arr_data,
                                               arr_centroids,
                                               arr_data_squares,
                                               arr_centroid_squares,
                                               block_size_in_rows,
                                               arr_responses,
                                               arr_distance_block,
                                               arr_closest_distances,
                                               { data_squares_event, centroid_squares_event });
        kernels_fp<Float>::compute_objective_function(queue,
                                                      arr_closest_distances,
                                                      arr_objective_function,
                                                      { assign_event })
            .wait_and_throw();

        return infer_result<task::clustering>()
            .set_responses(dal::homogen_table::wrap(arr_responses.flatten(queue), row_count, 1))
            .set_objective_function_value(
                static_cast<double>(*arr_objective_function.to_host(queue).get_data()));
    }
};

template struct infer_kernel_gpu<float, method::by_default, task::clustering>;
template struct infer_kernel_gpu<double, method::by_default, task::clustering>;

} // namespace oneapi::dal::kmeans::backend
