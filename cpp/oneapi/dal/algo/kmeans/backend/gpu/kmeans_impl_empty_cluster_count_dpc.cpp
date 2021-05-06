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

#include "oneapi/dal/algo/kmeans/backend/gpu/kmeans_impl.hpp"

namespace oneapi::dal::kmeans::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

struct partial_counters {};

struct merge_counters {};

sycl::event count_empty_clusters(sycl::queue& queue,
                                 std::int64_t cluster_count,
                                 pr::ndview<std::int32_t, 1>& counters,
                                 pr::ndarray<std::int32_t, 1>& empty_cluster_count,
                                 const bk::event_vector& deps) {
    ONEDAL_ASSERT(counters.get_dimension(0) == cluster_count);
    ONEDAL_ASSERT(empty_cluster_count.get_dimension(0) == 1);
    const std::int32_t* counter_ptr = counters.get_data();
    const auto sg_size_to_set = get_recommended_sg_size(queue);
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
