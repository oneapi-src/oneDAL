/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/algo/dbscan/backend/gpu/compute_kernel.hpp"
#include "oneapi/dal/algo/dbscan/backend/gpu/data_keeper.hpp"
#include "oneapi/dal/algo/dbscan/backend/gpu/results.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/backend/communicator.hpp"
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::dbscan::backend {

namespace bk = oneapi::dal::backend;
namespace pr = oneapi::dal::backend::primitives;
namespace spmd = oneapi::dal::preview::spmd;
namespace de = oneapi::dal::detail;

using dal::backend::context_gpu;

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;
using input_t = compute_input<task::clustering>;

template <typename Float>
static result_t compute_kernel_dense_impl(const context_gpu& ctx,
                                          const descriptor_t& desc,
                                          const table& local_data,
                                          const table& local_weights) {
    auto& comm = ctx.get_communicator();
    auto& queue = ctx.get_queue();

    std::int64_t rank_count = comm.get_rank_count();

    auto current_rank = comm.get_rank();

    auto prev_node = (current_rank - 1 + rank_count) % rank_count;
    auto next_node = (current_rank + 1) % rank_count;

    const std::int64_t local_row_count = local_data.get_row_count();
    const std::int64_t column_count = local_data.get_column_count();

    std::int64_t global_row_count = local_row_count;
    std::int64_t max_local_block_size = local_row_count;
    {
        ONEDAL_PROFILER_TASK(allreduce_rows_count_global, queue);
        comm.allreduce(max_local_block_size, spmd::reduce_op::max).wait();
    }

    auto global_rank_offsets = array<std::int64_t>::zeros(rank_count);
    global_rank_offsets.get_mutable_data()[current_rank] = local_row_count;
    {
        ONEDAL_PROFILER_TASK(allreduce_recv_counts, queue);
        comm.allreduce(global_rank_offsets, spmd::reduce_op::sum).wait();
    }
    {
        ONEDAL_PROFILER_TASK(allreduce_rows_count_global, queue);
        comm.allreduce(global_row_count, spmd::reduce_op::sum).wait();
    }

    std::int64_t local_offset = 0;

    for (std::int64_t i = 0; i < current_rank; i++) {
        ONEDAL_ASSERT(global_rank_offsets.get_data()[i] >= 0);
        local_offset += global_rank_offsets.get_data()[i];
    }

    const auto data_nd = pr::table2ndarray<Float>(queue, local_data, sycl::usm::alloc::device);
    pr::ndarray<Float, 2> data_nd_replace;
    if (rank_count > 1) {
        data_nd_replace = pr::ndarray<Float, 2>::empty(queue,
                                                       { max_local_block_size, column_count },
                                                       sycl::usm::alloc::device);

        copy(queue, data_nd_replace, data_nd, {}).wait_and_throw();
    }
    const auto weights_nd =
        pr::table2ndarray<Float>(queue, local_weights, sycl::usm::alloc::device);

    const Float epsilon = desc.get_epsilon() * desc.get_epsilon();
    const std::int64_t min_observations = desc.get_min_observations();

    //array indicates is the point core or no
    auto [arr_cores, cores_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, local_row_count, 0, sycl::usm::alloc::device);

    //array storages the information about neighbours of the point for distributed mode
    auto [arr_neighbours, neighbours_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, local_row_count, 0, sycl::usm::alloc::device);

    //array storages the information about neighbours of the point for distributed mode
    auto [arr_responses, responses_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, local_row_count, -1, sycl::usm::alloc::device);

    //array storages the information about which point are core neighbours in the current step
    auto [observation_indices, observation_indices_event] =
        pr::ndarray<bool, 1>::full(queue, local_row_count, false, sycl::usm::alloc::device);

    //array storages the information about count of the points in queue
    auto [queue_size, queue_size_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, 1, 0, sycl::usm::alloc::device);

    sycl::event::wait({ cores_event,
                        neighbours_event,
                        responses_event,
                        observation_indices_event,
                        queue_size_event });

    auto get_cores_event = kernels_fp<Float>::get_cores(queue,
                                                        data_nd,
                                                        weights_nd,
                                                        arr_cores,
                                                        arr_neighbours,
                                                        epsilon,
                                                        min_observations);

    for (std::int64_t j = 0; j < rank_count - 1; j++) {
        comm.sendrecv_replace(queue,
                              data_nd_replace.get_mutable_data(),
                              max_local_block_size * column_count,
                              prev_node,
                              next_node)
            .wait();
        auto local_row_block_count = global_rank_offsets.get_data()[next_node];
        auto actual_current_block = data_nd_replace.get_row_slice(0, local_row_block_count);
        kernels_fp<Float>::get_cores_send_recv_replace(queue,
                                                       data_nd,
                                                       actual_current_block,
                                                       weights_nd,
                                                       arr_cores,
                                                       arr_neighbours,
                                                       epsilon,
                                                       min_observations,
                                                       { get_cores_event })
            .wait_and_throw();
    }

    std::int64_t cluster_count = 0;

    std::int32_t cluster_index =
        kernels_fp<Float>::start_next_cluster(queue, arr_cores, arr_responses, { get_cores_event });
    cluster_index =
        cluster_index < local_row_count ? cluster_index + local_offset : global_row_count;
    {
        ONEDAL_PROFILER_TASK(allreduce_cluster_index, queue);
        comm.allreduce(cluster_index, spmd::reduce_op::min).wait();
    }

    if (cluster_index < 0) {
        return make_results(queue, desc, data_nd, arr_responses, arr_cores, 0, 0);
    }

    while (cluster_index < de::integral_cast<std::int32_t>(global_row_count)) {
        cluster_count++;

        bool in_range =
            cluster_index >= local_offset && cluster_index < local_offset + local_row_count;

        std::int32_t local_queue_size = 0;

        if (in_range) {
            set_arr_value(queue, arr_responses, cluster_index - local_offset, cluster_count - 1)
                .wait_and_throw();
            set_indices_in_area(queue, observation_indices, cluster_index - local_offset, true)
                .wait_and_throw();
            local_queue_size++;
        }

        std::int32_t total_queue_size = local_queue_size;

        {
            ONEDAL_PROFILER_TASK(allreduce_total_queue_size_outer, queue);
            comm.allreduce(total_queue_size, spmd::reduce_op::sum).wait();
        }

        while (total_queue_size != 0) {
            auto recv_counts = array<std::int64_t>::zeros(rank_count);
            recv_counts.get_mutable_data()[current_rank] = local_queue_size;
            {
                ONEDAL_PROFILER_TASK(allreduce_recv_counts, queue);
                comm.allreduce(recv_counts, spmd::reduce_op::sum).wait();
            }

            auto displs = array<std::int64_t>::zeros(rank_count);
            auto displs_ptr = displs.get_mutable_data();
            std::int64_t total_count = 0;
            for (std::int64_t i = 0; i < rank_count; i++) {
                displs_ptr[i] = total_count;
                total_count += recv_counts.get_data()[i];
            }

            auto [current_queue, current_queue_event] =
                pr::ndarray<Float, 2>::full(queue,
                                            { total_queue_size, column_count },
                                            0,
                                            sycl::usm::alloc::device);
            current_queue_event.wait_and_throw();
            kernels_fp<Float>::fill_current_queue(queue,
                                                  data_nd,
                                                  observation_indices,
                                                  current_queue,
                                                  displs_ptr[current_rank],
                                                  { current_queue_event })
                .wait_and_throw();

            set_arr_value(queue, queue_size, 0, total_queue_size).wait_and_throw();

            {
                ONEDAL_PROFILER_TASK(allreduce_xtx, queue);
                comm.allreduce(current_queue.flatten(queue, {}), spmd::reduce_op::sum).wait();
            }

            kernels_fp<Float>::update_queue(queue,
                                            data_nd,
                                            arr_cores,
                                            current_queue,
                                            arr_responses,
                                            queue_size,
                                            observation_indices,
                                            epsilon,
                                            cluster_count - 1)
                .wait_and_throw();

            local_queue_size = kernels_fp<Float>::get_queue_size(queue, queue_size);

            total_queue_size = local_queue_size;

            {
                ONEDAL_PROFILER_TASK(allreduce_total_queue_size_inner, queue);
                comm.allreduce(total_queue_size, spmd::reduce_op::sum).wait();
            }
        }

        cluster_index = kernels_fp<Float>::start_next_cluster(queue, arr_cores, arr_responses);
        cluster_index =
            cluster_index < local_row_count ? cluster_index + local_offset : global_row_count;
        {
            ONEDAL_PROFILER_TASK(cluster_index, queue);
            comm.allreduce(cluster_index, spmd::reduce_op::min).wait();
        }
    }

    return make_results(queue, desc, data_nd, arr_responses, arr_cores, cluster_count);
}

template <typename Float>
static result_t compute(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    return compute_kernel_dense_impl<Float>(ctx, desc, input.get_data(), input.get_weights());
}

template <typename Float>
struct compute_kernel_gpu<Float, method::brute_force, task::clustering> {
    result_t operator()(const context_gpu& ctx,
                        const descriptor_t& desc,
                        const input_t& input) const {
        return compute<Float>(ctx, desc, input);
    }
};

template struct compute_kernel_gpu<float, method::brute_force, task::clustering>;
template struct compute_kernel_gpu<double, method::brute_force, task::clustering>;

} // namespace oneapi::dal::dbscan::backend
