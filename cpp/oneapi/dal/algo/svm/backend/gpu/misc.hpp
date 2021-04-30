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

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/sort/sort.hpp"

namespace oneapi::dal::svm::backend {

namespace pr = dal::backend::primitives;

enum class ws_edge { up, low };

inline sycl::event arrange(sycl::queue& queue,
                           pr::ndview<std::uint32_t, 1>& indices_sort,
                           const std::int64_t n,
                           const dal::backend::event_vector& deps = {}) {
    ONEDAL_ASSERT(indices_sort.get_dimension(0) == n);
    ONEDAL_ASSERT(indices_sort.has_mutable_data());

    std::uint32_t* indices_sort_ptr = indices_sort.get_mutable_data();

    const auto wg_size = std::min(dal::backend::propose_wg_size(queue), n);
    const auto range = dal::backend::make_multiple_nd_range_1d(n, wg_size);

    auto arrange_event = queue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);

        cgh.parallel_for(range, [=](sycl::nd_item<1> item) {
            const std::uint32_t i = item.get_global_id(0);
            if (i < n) {
                indices_sort_ptr[i] = i;
            }
        });
    });

    return arrange_event;
}

template <typename Float>
inline sycl::event arg_sort(sycl::queue& queue,
                            const pr::ndview<Float, 1>& f,
                            pr::ndview<Float, 1>& values,
                            pr::ndview<std::uint32_t, 1>& indices_sort,
                            const std::int64_t n,
                            const dal::backend::event_vector& deps = {}) {
    ONEDAL_ASSERT(f.get_dimension(0) == n);
    ONEDAL_ASSERT(values.get_dimension(0) == n);
    ONEDAL_ASSERT(values.has_mutable_data());

    const Float* f_ptr = f.get_data();
    Float* values_ptr = values.get_mutable_data();

    auto copy_event = dal::backend::copy(queue, values_ptr, f_ptr, n, deps);
    auto arrange_event = arrange(queue, indices_sort, n);
    auto radix_sort = pr::radix_sort_indices_inplace<Float, std::uint32_t>{ queue };
    auto radix_sort_event = radix_sort(values, indices_sort, { copy_event, arrange_event });

    return radix_sort_event;
}

template <typename Float>
inline sycl::event check_ws_edge(sycl::queue& queue,
                                 const pr::ndview<Float, 1>& y,
                                 const pr::ndview<Float, 1>& alpha,
                                 pr::ndview<std::uint8_t, 1>& indicator,
                                 const Float C,
                                 const std::int64_t n,
                                 ws_edge edge,
                                 const dal::backend::event_vector& deps = {}) {
    ONEDAL_ASSERT(y.get_dimension(0) == n);
    ONEDAL_ASSERT(alpha.get_dimension(0) == n);
    ONEDAL_ASSERT(indicator.get_dimension(0) == n);
    ONEDAL_ASSERT(indicator.has_mutable_data());

    const Float* y_ptr = y.get_data();
    const Float* alpha_ptr = alpha.get_data();
    std::uint8_t* indicator_ptr = indicator.get_mutable_data();

    const auto wg_size = std::min(dal::backend::propose_wg_size(queue), n);
    const auto range = dal::backend::make_multiple_nd_range_1d(n, wg_size);

    sycl::event check_event;

    if (edge == ws_edge::up) {
        check_event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);

            cgh.parallel_for(range, [=](sycl::nd_item<1> item) {
                const std::uint32_t i = item.get_global_id(0);
                indicator_ptr[i] =
                    (y_ptr[i] > 0 && alpha_ptr[i] < C) || (y_ptr[i] < 0 && alpha_ptr[i] > 0);
            });
        });
    }
    else {
        check_event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);

            cgh.parallel_for(range, [=](sycl::nd_item<1> item) {
                const std::uint32_t i = item.get_global_id(0);
                indicator_ptr[i] =
                    (y_ptr[i] > 0 && alpha_ptr[i] > 0) || (y_ptr[i] < 0 && alpha_ptr[i] < C);
            });
        });
    }
    return check_event;
}

} // namespace oneapi::dal::svm::backend
