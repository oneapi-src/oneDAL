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

#pragma once

#include "oneapi/dal/algo/dbscan/backend/gpu/kernels_fp.hpp"
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::dbscan::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

inline std::int64_t get_recommended_sg_size(const sycl::queue& queue,
                                            std::int64_t column_count = 0) {
    // TODO optimization/dispatching
    return column_count > 32 ? 32 : 16;
}

template <typename Float, bool use_weights>
struct get_core_wide_kernel {
    static auto run(sycl::queue& queue,
                    const pr::ndview<Float, 2>& data,
                    const pr::ndview<Float, 2>& weights,
                    pr::ndview<std::int32_t, 1>& cores,
                    Float epsilon,
                    std::int64_t min_observations,
                    std::int64_t block_start,
                    std::int64_t block_end,
                    const bk::event_vector& deps) {
        using count_type = typename std::conditional<use_weights, Float, std::int64_t>::type;
        const auto row_count = data.get_dimension(0);
        ONEDAL_ASSERT(row_count > 0);
        ONEDAL_ASSERT(!use_weights || weights.get_dimension(0) == row_count);
        ONEDAL_ASSERT(!use_weights || weights.get_dimension(1) == 1);
        block_start = sycl::max(block_start, std::int64_t(0));
        block_end = (block_end < 0 || block_end > row_count) ? row_count : block_end;
        ONEDAL_ASSERT(block_start < block_end);
        const auto block_size = block_end - block_start;
        ONEDAL_ASSERT(cores.get_dimension(0) >= block_size);
        const std::int64_t column_count = data.get_dimension(1);

        const Float* data_ptr = data.get_data();
        const Float* weights_ptr = weights.get_data();
        std::int32_t* cores_ptr = cores.get_mutable_data();
        auto event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            std::int64_t wg_size = get_recommended_sg_size(queue, column_count);
            cgh.parallel_for(
                bk::make_multiple_nd_range_2d({ wg_size, block_size }, { wg_size, 1 }),
                [=](sycl::nd_item<2> item) {
                    auto sg = item.get_sub_group();
                    const std::uint32_t sg_id = sg.get_group_id()[0];
                    if (sg_id > 0)
                        return;
                    const std::uint32_t wg_id = item.get_global_id(1);
                    if (wg_id >= block_size)
                        return;
                    const std::uint32_t local_id = sg.get_local_id();
                    const std::uint32_t local_size = sg.get_local_range()[0];

                    count_type count = 0;
                    for (std::int64_t j = 0; j < row_count; j++) {
                        Float sum = Float(0);
                        for (std::int64_t i = local_id; i < column_count; i += local_size) {
                            Float val = data_ptr[(block_start + wg_id) * column_count + i] -
                                        data_ptr[j * column_count + i];
                            sum += val * val;
                        }
                        Float distance =
                            sycl::reduce_over_group(sg, sum, sycl::ext::oneapi::plus<Float>());
                        if constexpr (use_weights) {
                            count += distance <= epsilon ? weights_ptr[j] : count_type(0);
                        }
                        else {
                            count += distance <= epsilon ? count_type(1) : count_type(0);
                        }
                    }
                    if (local_id == 0) {
                        cores_ptr[wg_id] = (std::int32_t)(count >= min_observations);
                    }
                });
        });
        return event;
    }
    static constexpr std::int64_t min_width = 4;
};

template <typename Float, bool use_weights>
struct get_core_narrow_kernel {
    static auto run(sycl::queue& queue,
                    const pr::ndview<Float, 2>& data,
                    const pr::ndview<Float, 2>& weights,
                    pr::ndview<std::int32_t, 1>& cores,
                    Float epsilon,
                    std::int64_t min_observations,
                    std::int64_t block_start,
                    std::int64_t block_end,
                    const bk::event_vector& deps) {
        using count_type = typename std::conditional<use_weights, Float, std::int64_t>::type;
        const auto row_count = data.get_dimension(0);
        ONEDAL_ASSERT(row_count > 0);
        ONEDAL_ASSERT(!use_weights || weights.get_dimension(0) == row_count);
        ONEDAL_ASSERT(!use_weights || weights.get_dimension(1) == 1);
        block_start = sycl::max(block_start, std::int64_t(0));
        block_end = (block_end < 0 || block_end > row_count) ? row_count : block_end;
        ONEDAL_ASSERT(block_start < block_end);
        const auto block_size = block_end - block_start;
        ONEDAL_ASSERT(cores.get_dimension(0) >= block_size);
        const std::int64_t column_count = data.get_dimension(1);

        const Float* data_ptr = data.get_data();
        const Float* weights_ptr = weights.get_data();
        std::int32_t* cores_ptr = cores.get_mutable_data();
        auto event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);
            cgh.parallel_for(sycl::range<1>{ std::size_t(block_size) }, [=](sycl::id<1> idx) {
                count_type count = 0;
                for (std::int64_t j = 0; j < row_count; j++) {
                    Float sum = 0.0;
                    for (std::int64_t i = 0; i < column_count; i++) {
                        Float val = data_ptr[(block_start + idx) * column_count + i] -
                                    data_ptr[j * column_count + i];
                        sum += val * val;
                    }
                    if constexpr (use_weights) {
                        count += sum <= epsilon ? weights_ptr[j] : count_type(0);
                    }
                    else {
                        count += sum <= epsilon ? count_type(1) : count_type(0);
                    }
                }
                cores_ptr[idx] = (std::int32_t)(count >= min_observations);
            });
        });
        return event;
    }
    static constexpr std::int64_t max_width = 4;
};

template <typename Float>
template <bool use_weights>
sycl::event kernels_fp<Float>::get_cores_impl(sycl::queue& queue,
                                              const pr::ndview<Float, 2>& data,
                                              const pr::ndview<Float, 2>& weights,
                                              pr::ndview<std::int32_t, 1>& cores,
                                              Float epsilon,
                                              std::int64_t min_observations,
                                              std::int64_t block_start,
                                              std::int64_t block_end,
                                              const bk::event_vector& deps) {
    const std::int64_t column_count = data.get_dimension(1);
    if (column_count > get_core_wide_kernel<Float, use_weights>::min_width) {
        return get_core_wide_kernel<Float, use_weights>::run(queue,
                                                             data,
                                                             weights,
                                                             cores,
                                                             epsilon,
                                                             min_observations,
                                                             block_start,
                                                             block_end,
                                                             deps);
    }
    else {
        return get_core_narrow_kernel<Float, use_weights>::run(queue,
                                                               data,
                                                               weights,
                                                               cores,
                                                               epsilon,
                                                               min_observations,
                                                               block_start,
                                                               block_end,
                                                               deps);
    }
}

template <typename Float>
sycl::event kernels_fp<Float>::get_cores(sycl::queue& queue,
                                         const pr::ndview<Float, 2>& data,
                                         const pr::ndview<Float, 2>& weights,
                                         pr::ndview<std::int32_t, 1>& cores,
                                         Float epsilon,
                                         std::int64_t min_observations,
                                         std::int64_t block_start,
                                         std::int64_t block_end,
                                         const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(get_cores, queue);
    if (weights.get_dimension(0) == data.get_dimension(0)) {
        return get_cores_impl<true>(queue,
                                    data,
                                    weights,
                                    cores,
                                    epsilon,
                                    min_observations,
                                    block_start,
                                    block_end,
                                    deps);
    }
    return get_cores_impl<false>(queue,
                                 data,
                                 weights,
                                 cores,
                                 epsilon,
                                 min_observations,
                                 block_start,
                                 block_end,
                                 deps);
}

template <typename Float>
std::int32_t kernels_fp<Float>::start_next_cluster(sycl::queue& queue,
                                                   const pr::ndview<std::int32_t, 1>& cores,
                                                   pr::ndview<std::int32_t, 1>& responses,
                                                   const bk::event_vector& deps) {
    using oneapi::dal::backend::operator+;
    ONEDAL_PROFILER_TASK(start_next_cluster, queue);
    ONEDAL_ASSERT(cores.get_dimension(0) > 0);
    ONEDAL_ASSERT(cores.get_dimension(0) == responses.get_dimension(0));
    std::int64_t block_size = cores.get_dimension(0);

    auto [start_index, start_index_event] =
        pr::ndarray<std::int32_t, 1>::full(queue, { 1 }, block_size, sycl::usm::alloc::device);
    auto start_index_ptr = start_index.get_mutable_data();

    const std::int32_t* cores_ptr = cores.get_data();
    std::int32_t* responses_ptr = responses.get_mutable_data();
    std::int64_t wg_size = get_recommended_sg_size(queue);
    auto full_deps = deps + bk::event_vector{ start_index_event };
    auto index_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(full_deps);
        cgh.parallel_for(
            bk::make_multiple_nd_range_2d({ wg_size, 1 }, { wg_size, 1 }),
            [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::uint32_t sg_id = sg.get_group_id()[0];
                if (sg_id > 0)
                    return;
                const std::int32_t local_id = sg.get_local_id();
                const std::int32_t local_size = sg.get_local_range()[0];
                std::int32_t adjusted_block_size =
                    local_size * (block_size / local_size + bool(block_size % local_size));

                for (std::int32_t i = local_id; i < adjusted_block_size; i += local_size) {
                    const bool found =
                        i < block_size ? cores_ptr[i] == 1 && responses_ptr[i] < 0 : false;
                    const std::int32_t index =
                        sycl::reduce_over_group(sg,
                                                (std::int32_t)(found ? i : block_size),
                                                sycl::ext::oneapi::minimum<std::int32_t>());
                    if (index < block_size) {
                        if (local_id == 0) {
                            *start_index_ptr = index;
                        }
                        break;
                    }
                }
            });
    });
    return start_index.to_host(queue, { index_event }).at(0);
}

sycl::event set_queue_ptr(sycl::queue& queue,
                          pr::ndview<std::int32_t, 1>& algo_queue,
                          pr::ndview<std::int32_t, 1>& queue_front,
                          std::int32_t start_index,
                          const bk::event_vector& deps) {
    ONEDAL_ASSERT(queue_front.get_dimension(0) == 1);
    auto queue_ptr = algo_queue.get_mutable_data();
    auto queue_front_ptr = queue_front.get_mutable_data();

    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(bk::make_multiple_nd_range_2d({ 1, 1 }, { 1, 1 }),
                         [=](sycl::nd_item<2> item) {
                             queue_ptr[queue_front_ptr[0]] = start_index;
                             queue_front_ptr[0]++;
                         });
    });
}

sycl::event set_arr_value(sycl::queue& queue,
                          pr::ndview<std::int32_t, 1>& arr,
                          std::int32_t offset,
                          std::int32_t value,
                          const bk::event_vector& deps) {
    auto arr_ptr = arr.get_mutable_data();

    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for(bk::make_multiple_nd_range_2d({ 1, 1 }, { 1, 1 }),
                         [=](sycl::nd_item<2> item) {
                             arr_ptr[offset] = value;
                         });
    });
}

template <typename Float>
sycl::event kernels_fp<Float>::update_queue(sycl::queue& queue,
                                            const pr::ndview<Float, 2>& data,
                                            const pr::ndview<std::int32_t, 1>& cores,
                                            pr::ndview<std::int32_t, 1>& algo_queue,
                                            std::int32_t queue_begin,
                                            std::int32_t queue_end,
                                            pr::ndview<std::int32_t, 1>& responses,
                                            pr::ndview<std::int32_t, 1>& queue_front,
                                            Float epsilon,
                                            std::int32_t cluster_id,
                                            std::int64_t block_start,
                                            std::int64_t block_end,
                                            const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(update_algo_queue, queue);
    const auto row_count = data.get_dimension(0);
    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(queue_begin < algo_queue.get_dimension(0));
    ONEDAL_ASSERT(queue_end <= algo_queue.get_dimension(0));
    ONEDAL_ASSERT(queue_begin >= 0);
    ONEDAL_ASSERT(queue_end >= 0);
    ONEDAL_ASSERT(queue_front.get_dimension(0) == 1);
    block_start = (block_start < 0) ? 0 : block_start;
    block_end = (block_end < 0 || block_end > row_count) ? row_count : block_end;
    ONEDAL_ASSERT(block_start >= 0 && block_end > 0);
    ONEDAL_ASSERT(block_start < row_count && block_end <= row_count);
    const auto block_size = block_end - block_start;
    ONEDAL_ASSERT(cores.get_dimension(0) >= block_size);
    ONEDAL_ASSERT(responses.get_dimension(0) >= block_size);
    const std::int64_t column_count = data.get_dimension(1);
    ONEDAL_ASSERT(column_count > 0);
    const std::int32_t algo_queue_size = queue_end - queue_begin;

    const Float* data_ptr = data.get_data();
    std::int32_t* cores_ptr = cores.get_mutable_data();
    std::int32_t* queue_ptr = algo_queue.get_mutable_data();
    std::int32_t* queue_front_ptr = queue_front.get_mutable_data();
    std::int32_t* responses_ptr = responses.get_mutable_data();
    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        std::int64_t wg_size = get_recommended_sg_size(queue, column_count);
        cgh.parallel_for(
            bk::make_multiple_nd_range_2d({ wg_size, block_size }, { wg_size, 1 }),
            [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::uint32_t sg_id = sg.get_group_id()[0];
                if (sg_id > 0)
                    return;
                const std::uint32_t wg_id = item.get_global_id(1);
                if (wg_id >= block_size)
                    return;
                const std::uint32_t local_id = sg.get_local_id();
                const std::uint32_t local_size = sg.get_local_range()[0];
                const std::int32_t probe = block_start + wg_id;
                if (responses_ptr[wg_id] >= 0)
                    return;

                for (std::int32_t j = 0; j < algo_queue_size; j++) {
                    const std::int32_t index = queue_ptr[j + queue_begin];
                    Float sum = Float(0);
                    for (std::int64_t i = local_id; i < column_count; i += local_size) {
                        Float val =
                            data_ptr[probe * column_count + i] - data_ptr[index * column_count + i];
                        sum += val * val;
                    }
                    Float distance =
                        sycl::reduce_over_group(sg, sum, sycl::ext::oneapi::plus<Float>());
                    if (distance > epsilon)
                        continue;
                    if (local_id == 0) {
                        responses_ptr[wg_id] = cluster_id;
                    }

                    if (cores_ptr[wg_id] == 0)
                        continue;
                    if (local_id == 0) {
                        sycl::atomic_ref<std::int32_t,
                                         sycl::memory_order::relaxed,
                                         sycl::memory_scope::device,
                                         sycl::access::address_space::ext_intel_global_device_space>
                            counter_atomic(queue_front_ptr[0]);
                        std::int32_t new_front = counter_atomic.fetch_add(1);
                        queue_ptr[new_front] = probe;
                    }
                    break;
                }
            });
    });
    return event;
}

template <typename Float>
std::int32_t kernels_fp<Float>::get_queue_front(sycl::queue& queue,
                                                const pr::ndarray<std::int32_t, 1>& queue_front,
                                                const bk::event_vector& deps) {
    ONEDAL_ASSERT(queue_front.get_dimension(0) == 1);
    return queue_front.to_host(queue, deps).get_data()[0];
}

std::int64_t count_cores(sycl::queue& queue, const pr::ndview<std::int32_t, 1>& cores) {
    const std::uint64_t row_count = cores.get_dimension(0);
    ONEDAL_ASSERT(row_count > 0);
    std::int64_t sum_result = 0;
    auto cores_ptr = cores.get_data();
    sycl::buffer<std::int64_t> sum_buf{ &sum_result, 1 };
    queue
        .submit([&](sycl::handler& cgh) {
            auto sum_reduction = reduction(sum_buf, cgh, sycl::ext::oneapi::plus<std::int64_t>());
            cgh.parallel_for(sycl::range<1>{ row_count },
                             sum_reduction,
                             [=](sycl::id<1> idx, auto& sum) {
                                 sum.combine(cores_ptr[idx]);
                             });
        })
        .wait_and_throw();
    return sum_buf.get_host_access()[0];
}

#endif

} // namespace oneapi::dal::dbscan::backend
