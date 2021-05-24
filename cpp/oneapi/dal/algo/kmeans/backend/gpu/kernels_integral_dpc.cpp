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

#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/distance.hpp"
#include "oneapi/dal/backend/primitives/sort/sort.hpp"

namespace oneapi::dal::kmeans::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

inline std::int64_t get_recommended_sg_size2(sycl::queue& queue) {
    // TODO optimization/dispatching
    return 16;
}

inline std::int64_t get_recommended_wg_count2(sycl::queue& queue) {
    // TODO optimization/dispatching
    return 128;
}

struct partial_counters {};

struct merge_counters {};

sycl::event count_clusters(sycl::queue& queue,
                           const pr::ndview<std::int32_t, 2>& labels,
                           std::int64_t cluster_count,
                           pr::ndview<std::int32_t, 1>& counters,
                           const bk::event_vector& deps) {
    ONEDAL_ASSERT(counters.get_dimension(0) == cluster_count);
    ONEDAL_ASSERT(labels.get_dimension(1) == 1);
    const std::int32_t* label_ptr = labels.get_data();
    std::int32_t* counter_ptr = counters.get_mutable_data();
    const auto sg_size_to_set = get_recommended_sg_size2(queue);
    const auto wg_count_to_set = get_recommended_wg_count2(queue);
    queue
        .submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(cluster_count), [=](sycl::id<1> idx) {
                counter_ptr[idx] = 0;
            });
        })
        .wait_and_throw();
    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        const auto row_count = labels.get_dimension(0);
        cgh.parallel_for<partial_counters>(
            bk::make_multiple_nd_range_2d({ sg_size_to_set, wg_count_to_set },
                                          { sg_size_to_set, 1 }),
            [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::int64_t sg_id = sg.get_group_id()[0];
                const std::int64_t wg_id = item.get_global_id(1);
                const std::int64_t wg_count = item.get_global_range(1);
                const std::int64_t sg_count = sg.get_group_range()[0];
                const std::int64_t sg_global_id = wg_id * sg_count + sg_id;
                const std::int64_t total_sg_count = wg_count * sg_count;

                const std::int64_t local_id = sg.get_local_id()[0];
                const std::int64_t local_range = sg.get_local_range()[0];

                const std::int64_t block_size =
                    row_count / total_sg_count + std::int64_t(row_count % total_sg_count > 0);
                const std::int64_t offset = block_size * sg_global_id;
                const std::int64_t end =
                    (offset + block_size) > row_count ? row_count : (offset + block_size);
                for (std::int64_t i = offset + local_id; i < end; i += local_range) {
                    const std::int32_t cl = label_ptr[i];
                    sycl::ONEAPI::atomic_ref<std::int32_t,
                                             cl::sycl::ONEAPI::memory_order::relaxed,
                                             cl::sycl::ONEAPI::memory_scope::device,
                                             cl::sycl::access::address_space::global_device_space>
                        counter_atomic(counter_ptr[cl]);
                    counter_atomic.fetch_add(1);
                }
            });
    });
}

sycl::event count_empty_clusters(sycl::queue& queue,
                                 std::int64_t cluster_count,
                                 pr::ndview<std::int32_t, 1>& counters,
                                 pr::ndarray<std::int32_t, 1>& empty_cluster_count,
                                 const bk::event_vector& deps) {
    ONEDAL_ASSERT(counters.get_dimension(0) == cluster_count);
    ONEDAL_ASSERT(empty_cluster_count.get_dimension(0) == 1);
    const std::int32_t* counter_ptr = counters.get_data();
    const auto sg_size_to_set = get_recommended_sg_size2(queue);
    std::int32_t* value_ptr = empty_cluster_count.get_mutable_data();
    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for<merge_counters>(
            bk::make_multiple_nd_range_2d({ sg_size_to_set, 1 }, { sg_size_to_set, 1 }),
            [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::int64_t sg_id = sg.get_group_id()[0];
                if (sg_id > 0)
                    return;
                const std::int64_t local_id = sg.get_local_id()[0];
                const std::int64_t local_range = sg.get_local_range()[0];
                std::int32_t sum = 0;
                for (std::int64_t i = local_id; i < cluster_count; i += local_range) {
                    sum += counter_ptr[i] == 0;
                }
                sum = reduce(sg, sum, sycl::ONEAPI::plus<std::int32_t>());
                if (local_id == 0) {
                    value_ptr[0] = sum;
                }
            });
    });
}
#endif

} // namespace oneapi::dal::kmeans::backend
