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

inline std::int64_t get_scaled_wg_size_per_row(const sycl::queue& queue,
                                               std::int64_t column_count,
                                               std::int64_t preffered_wg_size) {
    const std::int64_t sg_max_size = bk::device_max_sg_size(queue);
    const std::int64_t row_adjusted_sg_num =
        column_count / sg_max_size + std::int64_t(column_count % sg_max_size > 0);
    std::int64_t expected_sg_num = std::min(preffered_wg_size / sg_max_size, row_adjusted_sg_num);
    if (expected_sg_num < 1)
        expected_sg_num = 1;
    return dal::detail::check_mul_overflow(expected_sg_num, sg_max_size);
}

template<typename T>
struct select_min_distance {};

template <typename Float>
sycl::event select(sycl::queue& queue,
                   const pr::ndview<Float, 2>& data,
                   pr::ndview<Float, 2>& selection,
                   pr::ndview<std::int32_t, 2>& indices,
                   const bk::event_vector& deps = {}) {
    ONEDAL_ASSERT(!indices_out || indices.get_shape()[0] == data.get_shape()[0]);
    ONEDAL_ASSERT(!indices_out || indices.get_shape()[1] == 1);
    ONEDAL_ASSERT(!selection_out || selection.get_shape()[0] == data.get_shape()[0]);
    ONEDAL_ASSERT(!selection_out || selection.get_shape()[1] == 1);

    const std::int64_t col_count = data.get_dimension(1);
    const std::int64_t row_count = data.get_dimension(0);
    const std::int64_t stride = data.get_shape()[1];

    const std::int64_t preffered_wg_size = 128;
    const std::int64_t wg_size = get_scaled_wg_size_per_row(queue, col_count, preffered_wg_size);

    const Float* data_ptr = data.get_data();
    Float* selection_ptr = selection.get_mutable_data();
    std::int32_t* indices_ptr = indices.get_mutable_data();
    const auto fp_max = detail::limits<Float>::max();

    auto event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for<select_min_distance<Float>>(
            bk::make_multiple_nd_range_2d({ wg_size, row_count }, { wg_size, 1 }),
            [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::uint32_t sg_id = sg.get_group_id()[0];
                const std::uint32_t wg_id = item.get_global_id(1);
                const std::uint32_t sg_num = sg.get_group_range()[0];
                const std::uint32_t sg_global_id = wg_id * sg_num + sg_id;
                if (sg_global_id >= row_count)
                    return;
                const std::uint32_t in_offset = sg_global_id * stride;
                const std::uint32_t out_offset = sg_global_id;

                const std::uint32_t local_id = sg.get_local_id()[0];
                const std::uint32_t local_range = sg.get_local_range()[0];

                std::int32_t index = -1;
                Float value = fp_max;
                for (std::uint32_t i = local_id; i < col_count; i += local_range) {
                    const Float cur_val = data_ptr[in_offset + i];
                    if (cur_val < value) {
                        index = i;
                        value = cur_val;
                    }
                }

                sg.barrier();

                const Float final_value = reduce(sg, value, sycl::ONEAPI::minimum<Float>());
                const bool present = (final_value == value);
                const std::int32_t pos =
                    exclusive_scan(sg, present ? 1 : 0, sycl::ONEAPI::plus<std::int32_t>());
                const bool owner = present && pos == 0;
                const std::int32_t final_index =
                    -reduce(sg, owner ? -index : 1, sycl::ONEAPI::minimum<std::int32_t>());

                if (local_id == 0) {
                    indices_ptr[out_offset] = final_index;
                    selection_ptr[out_offset] = final_value;
                }
            });
    });
    return event;
}

template <typename Float, typename Metric>
sycl::event assign_clusters(sycl::queue& queue,
                            const pr::ndview<Float, 2>& data,
                            const pr::ndview<Float, 2>& centroids,
                            std::int64_t block_rows,
                            pr::ndview<std::int32_t, 2>& labels,
                            pr::ndview<Float, 2>& distances,
                            pr::ndview<Float, 2>& closest_distances,
                            const bk::event_vector& deps) {
    ONEDAL_ASSERT(data.get_shape()[1] == centroids.get_shape()[1]);
    ONEDAL_ASSERT(data.get_shape()[0] >= centroids.get_shape()[0]);
    ONEDAL_ASSERT(labels.get_shape()[0] >= data.get_shape()[0]);
    ONEDAL_ASSERT(labels.get_shape()[1] == 1);
    ONEDAL_ASSERT(closest_distances.get_shape()[0] >= data.get_shape()[0]);
    ONEDAL_ASSERT(closest_distances.get_shape()[1] == 1);
    ONEDAL_ASSERT(distances.get_shape()[0] >= block_rows);
    ONEDAL_ASSERT(distances.get_shape()[1] >= centroids.get_shape()[0]);
    sycl::event selection_event;
    auto row_count = data.get_shape()[0];
    auto column_count = data.get_shape()[1];
    auto centroid_count = centroids.get_shape()[0];
    pr::distance<Float, Metric> block_distances(queue);
    auto block_count = row_count / block_rows + std::int64_t(row_count % block_rows > 0);
    for (std::int64_t iblock = 0; iblock < block_count; iblock++) {
        auto row_offset = block_rows * iblock;
        auto cur_rows = std::min(block_rows, row_count - row_offset);
        auto distance_block =
            pr::ndview<Float, 2>::wrap(distances.get_mutable_data(), { cur_rows, centroid_count });
        auto data_block = pr::ndview<Float, 2>::wrap(data.get_data() + row_offset * column_count,
                                                     { cur_rows, column_count });
        auto distance_event =
            block_distances(data_block, centroids, distance_block, { selection_event });
        auto label_block =
            pr::ndview<int32_t, 2>::wrap(labels.get_mutable_data() + row_offset, { cur_rows, 1 });
        auto closest_distance_block =
            pr::ndview<Float, 2>::wrap(closest_distances.get_mutable_data() + row_offset,
                                       { cur_rows, 1 });
        selection_event =
            select(queue, distance_block, closest_distance_block, label_block, { distance_event });
    }
    return selection_event;
}

#define INSTANTIATE_WITH_METRIC(F, M)                                                  \
    template sycl::event assign_clusters<F, M<F>>(sycl::queue & queue,                 \
                                                  const pr::ndview<F, 2>& data,        \
                                                  const pr::ndview<F, 2>& centroids,   \
                                                  std::int64_t block_rows,             \
                                                  pr::ndview<std::int32_t, 2>& labels, \
                                                  pr::ndview<F, 2>& distances,         \
                                                  pr::ndview<F, 2>& closest_distances, \
                                                  const bk::event_vector& deps);

#define INSTANTIATE(F)                                                                            \
    INSTANTIATE_WITH_METRIC(F, pr::squared_l2_metric)

INSTANTIATE(float)
INSTANTIATE(double)

#endif

} // namespace oneapi::dal::kmeans::backend
