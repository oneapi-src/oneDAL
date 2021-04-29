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

#include "oneapi/dal/backend/primitives/selection/kselect_by_rows_single_col.hpp"
#include "oneapi/dal/backend/primitives/sort/sort.hpp"
#include "oneapi/dal/algo/kmeans/backend/gpu/kmeans_impl.hpp"

namespace oneapi::dal::kmeans::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;


static std::int64_t get_gpu_sg_size(sycl::queue& queue) {
    // TODO optimization/dispatching
    return 16;
}

template<typename T>
struct compute_obj_function {};

template <typename Float>
sycl::event compute_objective_function(sycl::queue& queue,
                                       const pr::ndview<Float, 2>& closest_distances,
                                       pr::ndview<Float, 1>& objective_function,
                                       const bk::event_vector& deps) {
    ONEDAL_ASSERT(closest_distances.get_shape()[1] == 1);
    ONEDAL_ASSERT(objective_function.get_shape()[0] == 1);
    const Float* distance_ptr = closest_distances.get_data();
    Float* value_ptr = objective_function.get_mutable_data();
    const auto row_count = closest_distances.get_shape()[0];
    const auto sg_size_to_set = get_gpu_sg_size(queue);
    return queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        cgh.parallel_for<compute_obj_function<Float>>(
            bk::make_multiple_nd_range_2d({ sg_size_to_set, 1 }, { sg_size_to_set, 1 }),
            [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const std::int64_t sg_id = sg.get_group_id()[0];
                if (sg_id > 0)
                    return;
                const std::int64_t local_id = sg.get_local_id()[0];
                const std::int64_t local_range = sg.get_local_range()[0];
                Float sum = 0;
                for (std::int64_t i = local_id; i < row_count; i += local_range) {
                    sum += distance_ptr[i];
                }
                sum = reduce(sg, sum, sycl::ONEAPI::plus<Float>());
                if (local_id == 0) {
                    value_ptr[0] = sum;
                }
            });
    });
}

#define INSTANTIATE(F)                                                                            \
    template sycl::event compute_objective_function<F>(sycl::queue & queue,                       \
                                                       const pr::ndview<F, 2>& closest_distances, \
                                                       pr::ndview<F, 1>& objective_function,      \
                                                       const bk::event_vector& deps);

INSTANTIATE(float)
INSTANTIATE(double)

#endif

} // namespace oneapi::dal::kmeans::backend
