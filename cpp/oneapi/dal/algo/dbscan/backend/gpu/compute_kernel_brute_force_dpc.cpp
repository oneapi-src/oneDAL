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
#include <limits>
#include <cmath>
#include <iostream>
#include <iomanip>

namespace bk = oneapi::dal::backend;
namespace pr = oneapi::dal::backend::primitives;
namespace spmd = oneapi::dal::preview::spmd;
namespace de = oneapi::dal::detail;

namespace oneapi::dal::dbscan::backend {

using dal::backend::context_gpu;

using descriptor_t = detail::descriptor_base<task::clustering>;
using result_t = compute_result<task::clustering>;
using input_t = compute_input<task::clustering>;

template <typename Type>
std::ostream& print_on_host(std::ostream& stream, const oneapi::dal::array<Type>& array) {
    const std::int64_t count = array.get_count();

    if (count < std::int64_t(1)) {
        stream << "An empty array" << std::endl;
    }

    constexpr std::int32_t precision = std::is_floating_point_v<Type> ? 3 : 0;

    stream << std::setw(10);
    stream << std::setprecision(precision);
    stream << std::setiosflags(std::ios::fixed);
    for (std::int64_t i = 0l; i < count; ++i) {
        stream << array[i] << ' ';
    }

    return stream;
}

template <typename Type>
std::ostream& operator<<(std::ostream& stream, const oneapi::dal::array<Type>& array) {
    oneapi::dal::array<Type> array_on_host = to_host(array);
    return print_on_host(stream, array_on_host);
}

std::ostream& operator<<(std::ostream& stream, const oneapi::dal::table& table) {
    auto arr = oneapi::dal::row_accessor<const float>(table).pull();
    const auto x = arr.get_data();

    if (true) {
        for (std::int64_t i = 0; i < table.get_row_count(); i++) {
            for (std::int64_t j = 0; j < table.get_column_count(); j++) {
                std::cout << std::setw(10) << std::setiosflags(std::ios::fixed)
                          << std::setprecision(3) << x[i * table.get_column_count() + j];
            }
            std::cout << std::endl;
        }
    }
    else {
        for (std::int64_t i = 0; i < 5; i++) {
            for (std::int64_t j = 0; j < table.get_column_count(); j++) {
                std::cout << std::setw(10) << std::setiosflags(std::ios::fixed)
                          << std::setprecision(3) << x[i * table.get_column_count() + j];
            }
            std::cout << std::endl;
        }
        std::cout << "..." << (table.get_row_count() - 10) << " lines skipped..." << std::endl;
        for (std::int64_t i = table.get_row_count() - 5; i < table.get_row_count(); i++) {
            for (std::int64_t j = 0; j < table.get_column_count(); j++) {
                std::cout << std::setw(10) << std::setiosflags(std::ios::fixed)
                          << std::setprecision(3) << x[i * table.get_column_count() + j];
            }
            std::cout << std::endl;
        }
    }
    return stream;
}

template <typename Float>
static result_t compute_kernel_dense_impl(const context_gpu& ctx,
                                          const descriptor_t& desc,
                                          const table& local_data,
                                          const table& local_weights) {
    auto& comm = ctx.get_communicator();
    auto& queue = ctx.get_queue();

    std::int64_t rank_count = comm.get_rank_count();

    auto current_rank = comm.get_rank();

    // auto prev_node = (current_rank - 1 + rank_count) % rank_count;
    // auto next_node = (current_rank + 1) % rank_count;

    const std::int64_t row_count = local_data.get_row_count();
    const std::int64_t column_count = local_data.get_column_count();
    std::int64_t global_row_count = row_count;

    auto global_rank_offsets = array<std::int64_t>::zeros(rank_count);
    global_rank_offsets.get_mutable_data()[current_rank] = row_count;
    {
        ONEDAL_PROFILER_TASK(allreduce_recv_counts, queue);
        comm.allreduce(global_rank_offsets, spmd::reduce_op::sum).wait();
    }
    {
        ONEDAL_PROFILER_TASK(allreduce_rows_count_global);
        comm.allreduce(global_row_count, spmd::reduce_op::sum).wait();
    }

    // std::int64_t local_offset = 0;

    // for (std::int64_t i = 0; i < current_rank; i++) {
    //     ONEDAL_ASSERT(global_rank_offsets.get_data()[i] >= 0);
    //     local_offset += global_rank_offsets.get_data()[i];
    // }
    std::unordered_map<int, int> element_to_id;
    // std::unordered_map<int, int> element_counts;
    std::unordered_map<int, int> element_to_id_final;

    int current_id = 0;
    int final_id = 0;

    const auto data_nd = pr::table2ndarray<Float>(queue, local_data, sycl::usm::alloc::device);

    auto data_nd_replace =
        pr::ndarray<Float, 2>::empty(queue, { row_count, column_count }, sycl::usm::alloc::device);
    if (rank_count > 1) {
        auto copy_event = copy(queue, data_nd_replace, data_nd, {});
        copy_event.wait_and_throw();
    }

    const auto weights_nd =
        pr::table2ndarray<Float>(queue, local_weights, sycl::usm::alloc::device);

    const double epsilon = desc.get_epsilon() * desc.get_epsilon();
    const std::int64_t min_observations = desc.get_min_observations();

    auto [arr_cores, cores_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, row_count, -1, sycl::usm::alloc::device);
    auto [arr_neighbours, neighbours_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, row_count, 0, sycl::usm::alloc::device);
    auto [arr_responses_fake, responses_fake_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, row_count, -1, sycl::usm::alloc::device);

    // auto arr_responses_ptr = arr_responses_fake.get_mutable_data();

    // auto init_event = queue.submit([&](sycl::handler& cgh) {
    //     const auto range = sycl::range<1>(global_row_count);

    //     cgh.parallel_for(range, [=](sycl::item<1> id) {
    //         arr_responses_ptr[id] = id;
    //     });
    // });

    // auto [arr_responses, responses_event] =
    //     pr::ndarray<std::int32_t, 1>::full(queue, row_count, -1, sycl::usm::alloc::device);
    // auto [arr_queue, queue_event] =
    //     pr::ndarray<std::int32_t, 1>::full(queue, global_row_count, -1, sycl::usm::alloc::device);
    auto [arr_queue_front, queue_front_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, 1, 0, sycl::usm::alloc::device);
    auto [adj_matrix, adj_matrix_event] =
        pr::ndarray<std::int32_t, 2>::full(queue,
                                           { row_count, global_row_count },
                                           0,
                                           sycl::usm::alloc::device);
    sycl::event::wait({ cores_event,
                        responses_fake_event,
                        queue_front_event,
                        adj_matrix_event,
                        neighbours_event });

    kernels_fp<Float>::get_cores(queue,
                                 data_nd,
                                 weights_nd,
                                 arr_cores,
                                 arr_responses_fake,
                                 arr_neighbours,
                                 adj_matrix,
                                 epsilon,
                                 min_observations)
        .wait_and_throw();
    // std::cout << adj_matrix << std::endl;
    // for (std::int64_t j = 0; j < rank_count - 1; j++) {
    //     comm.sendrecv_replace(queue,
    //                           data_nd_replace.get_mutable_data(),
    //                           row_count * column_count,
    //                           prev_node,
    //                           next_node)
    //         .wait();
    //     kernels_fp<Float>::get_cores_send_recv_replace(queue,
    //                                                    data_nd,
    //                                                    data_nd_replace,
    //                                                    weights_nd,
    //                                                    arr_cores,
    //                                                    arr_neighbours,
    //                                                    epsilon,
    //                                                    min_observations)
    //         .wait_and_throw();
    // }

    // auto arr_fake_responses_host = arr_responses_fake.to_host(queue);

    // auto fake_responses_ptr = arr_fake_responses_host.get_mutable_data();
    // std::cout << "after get cores components" << std::endl;
    // for (int i = 0; i < row_count; i++) {
    //     std::cout << fake_responses_ptr[i] << std::endl;
    // }
    bool flag = true;

    while (flag) {
        connected_components(queue,
                             arr_responses_fake,
                             adj_matrix,
                             arr_queue_front,
                             arr_neighbours,
                             min_observations)
            .wait_and_throw();
        auto value = kernels_fp<Float>::get_queue_front(queue, arr_queue_front);
        if (value == 0) {
            flag = false;
        }
    }
    auto arr_fake_responses_host = arr_responses_fake.to_host(queue);

    auto fake_responses_ptr = arr_fake_responses_host.get_mutable_data();

    for (int i = 0; i < row_count; i++) {
        if (fake_responses_ptr[i] != -1) {
            if (element_to_id.find(fake_responses_ptr[i]) == element_to_id.end()) {
                element_to_id[fake_responses_ptr[i]] = current_id++;
            }
        }
        //element_counts[fake_responses_ptr[i]]++;
    }

    for (int i = 0; i < row_count; i++) {
        if (fake_responses_ptr[i] != -1) {
            if (element_to_id_final.find(fake_responses_ptr[i]) == element_to_id_final.end()) {
                element_to_id_final[fake_responses_ptr[i]] = final_id++;
            }
        }
    }

    for (int i = 0; i < row_count; i++) {
        if (fake_responses_ptr[i] != -1) {
            fake_responses_ptr[i] = element_to_id_final[fake_responses_ptr[i]];
        }
    }
    auto arr_responses_ordered = arr_fake_responses_host.to_device(queue);

    return make_results(queue, desc, data_nd, arr_responses_ordered, arr_cores, final_id);
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
