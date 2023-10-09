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

#include "oneapi/dal/detail/profiler.hpp"

namespace bk = oneapi::dal::backend;
namespace pr = oneapi::dal::backend::primitives;
namespace spmd = oneapi::dal::preview::spmd;
namespace de = oneapi::dal::detail;

namespace oneapi::dal::dbscan::backend {

using dal::backend::context_gpu;

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;
using input_t = compute_input<task::clustering>;

template <typename Float>
static result_t call_daal_kernel(const context_gpu& ctx,
                                 const descriptor_t& desc,
                                 const table& local_data,
                                 const table& local_weights) {
    auto& comm = ctx.get_communicator();
    auto& queue = ctx.get_queue();

    data_keeper<Float> keeper(ctx);
    {
        ONEDAL_PROFILER_TASK(init_data, queue);
        keeper.init(local_data, local_weights);
    }
    const std::int64_t block_size = keeper.get_block_size();
    const std::int64_t block_start = 0;
    const std::int64_t block_end = block_start + block_size;
    const std::int64_t row_count = keeper.get_row_count();
    auto arr_data = pr::table2ndarray<Float>(queue, local_data, sycl::usm::alloc::device);
    auto arr_weights = pr::table2ndarray<Float>(queue, local_weights, sycl::usm::alloc::device);

    std::int64_t rank = comm.get_rank();
    std::int64_t rank_count = comm.get_rank_count();

    const double epsilon = desc.get_epsilon() * desc.get_epsilon();
    const std::int64_t min_observations = desc.get_min_observations();

    auto [arr_cores, cores_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, block_size, 0, sycl::usm::alloc::device);
    auto [arr_responses, responses_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, block_size, -1, sycl::usm::alloc::device);
    auto [arr_queue, queue_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, row_count, -1, sycl::usm::alloc::device);
    auto [arr_queue_front, queue_front_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, 1, 0, sycl::usm::alloc::device);

    sycl::event::wait({ cores_event, responses_event, queue_event, queue_front_event });

    kernels_fp<Float>::get_cores(queue,
                                 arr_data,
                                 arr_weights,
                                 arr_cores,
                                 epsilon,
                                 min_observations,
                                 block_start,
                                 block_size)
        .wait_and_throw();

    std::int64_t cluster_count = 0;
    std::int32_t cluster_index =
        kernels_fp<Float>::start_next_cluster(queue, arr_cores, arr_responses);
    cluster_index = cluster_index < block_size ? cluster_index + block_start : row_count;
    {
        ONEDAL_PROFILER_TASK(allreduce_cluster_index, queue);
        comm.allreduce(cluster_index, spmd::reduce_op::min).wait();
    }
    if (cluster_index < 0) {
        return make_results(queue, desc, arr_data, arr_responses, arr_cores, 0, 0);
    }
    std::int32_t queue_begin = 0;
    std::int32_t queue_end = 0;
    while (cluster_index < de::integral_cast<std::int32_t>(row_count)) {
        cluster_count++;
        bool in_range = cluster_index >= block_start && cluster_index < block_end;
        if (in_range) {
            set_arr_value(queue, arr_responses, cluster_index - block_start, cluster_count - 1)
                .wait_and_throw();
            set_queue_ptr(queue, arr_queue, arr_queue_front, cluster_index).wait_and_throw();
            queue_end++;
        }
        std::int32_t local_queue_size = queue_end - queue_begin;
        std::int32_t total_queue_size = local_queue_size;
        std::cout << "total_queue_size local init = " << total_queue_size << std::endl;
        {
            ONEDAL_PROFILER_TASK(allreduce_total_queue_size_outer, queue);
            comm.allreduce(total_queue_size, spmd::reduce_op::sum).wait();
        }
        std::cout << "total_queue_size global init = " << total_queue_size << std::endl;
        while (total_queue_size > 0) {
            auto recv_counts = array<std::int64_t>::zeros(rank_count);
            recv_counts.get_mutable_data()[rank] = local_queue_size;
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

            ONEDAL_ASSERT(total_count > 0);
            auto send_array = recv_counts[rank] > 0
                                  ? arr_queue.slice(queue_begin, recv_counts[rank]).flatten(queue)
                                  : array<std::int32_t>::wrap(queue, arr_queue.get_data(), 0);
            if (rank_count > 1 && recv_counts[rank] > 0) {
                auto [arr_copy, arr_event] =
                    pr::ndarray<std::int32_t, 1>::copy(queue,
                                                       arr_queue.get_data() + queue_begin,
                                                       recv_counts[rank],
                                                       sycl::usm::alloc::device);
                arr_event.wait_and_throw();
                send_array = arr_copy.flatten(queue);
            }
            auto recv_array = arr_queue.slice(queue_begin, total_count).flatten(queue);
            {
                ONEDAL_PROFILER_TASK(allgather_cluster_data, queue);
                comm.allgatherv(send_array, recv_array, recv_counts.get_data(), displs.get_data())
                    .wait();
            }
            queue_end = queue_begin + total_queue_size;
            arr_queue_front.fill(queue, queue_end).wait_and_throw();
            kernels_fp<Float>::update_queue(queue,
                                            arr_data,
                                            arr_cores,
                                            arr_queue,
                                            queue_begin,
                                            queue_end,
                                            arr_responses,
                                            arr_queue_front,
                                            epsilon,
                                            cluster_count - 1,
                                            block_start,
                                            block_end)
                .wait_and_throw();
            queue_begin = queue_end;
            queue_end = kernels_fp<Float>::get_queue_front(queue, arr_queue_front);
            local_queue_size = queue_end - queue_begin;
            total_queue_size = local_queue_size;
            std::cout << "total_queue_size global after update = " << total_queue_size << std::endl;
            {
                ONEDAL_PROFILER_TASK(allreduce_total_queue_size_inner, queue);
                comm.allreduce(total_queue_size, spmd::reduce_op::sum).wait();
            }
        }
        std::cout << "total_queue_size global after while = " << total_queue_size << std::endl;
        cluster_index = kernels_fp<Float>::start_next_cluster(queue, arr_cores, arr_responses);
        cluster_index = cluster_index < block_size ? cluster_index + block_start : row_count;
        {
            ONEDAL_PROFILER_TASK(cluster_index, queue);
            comm.allreduce(cluster_index, spmd::reduce_op::min).wait();
        }
    }
    return make_results(queue, desc, arr_data, arr_responses, arr_cores, cluster_count);
}

template <typename Float>
static result_t compute(const context_gpu& ctx, const descriptor_t& desc, const input_t& input) {
    return call_daal_kernel<Float>(ctx, desc, input.get_data(), input.get_weights());
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
