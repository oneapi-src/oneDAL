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

namespace oneapi::dal::svm::backend {

namespace pr = dal::backend::primitives;

enum class violating_edge { up, low };

template <typename Float>
inline bool is_upper_edge(const Float y, const Float alpha, const Float C) {
    return (y > 0 && alpha < C) || (y < 0 && alpha > 0);
}

template <typename Float>
inline bool is_lower_edge(const Float y, const Float alpha, const Float C) {
    return (y > 0 && alpha > 0) || (y < 0 && alpha < C);
}

template <typename Float>
sycl::event check_violating_edge(sycl::queue& queue,
                                 const pr::ndview<Float, 1>& y,
                                 const pr::ndview<Float, 1>& alpha,
                                 pr::ndview<std::uint8_t, 1>& indicator,
                                 const Float C,
                                 violating_edge edge,
                                 const dal::backend::event_vector& deps = {}) {
    ONEDAL_ASSERT(y.get_dimension(0) == alpha.get_dimension(0));
    ONEDAL_ASSERT(alpha.get_dimension(0) == indicator.get_dimension(0));
    ONEDAL_ASSERT(indicator.has_mutable_data());

    const Float* y_ptr = y.get_data();
    const Float* alpha_ptr = alpha.get_data();
    std::uint8_t* indicator_ptr = indicator.get_mutable_data();

    const std::int64_t row_count = y.get_dimension(0);

    const auto wg_size = std::min(dal::backend::propose_wg_size(queue), row_count);
    const auto range = dal::backend::make_multiple_nd_range_1d(row_count, wg_size);

    sycl::event check_event;

    if (edge == violating_edge::up) {
        check_event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);

            cgh.parallel_for(range, [=](sycl::nd_item<1> item) {
                const std::uint32_t i = item.get_global_id(0);
                indicator_ptr[i] = is_upper_edge<Float>(y_ptr[i], alpha_ptr[i], C);
            });
        });
    }
    else {
        check_event = queue.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);

            cgh.parallel_for(range, [=](sycl::nd_item<1> item) {
                const std::uint32_t i = item.get_global_id(0);
                indicator_ptr[i] = is_lower_edge<Float>(y_ptr[i], alpha_ptr[i], C);
            });
        });
    }
    return check_event;
}

} // namespace oneapi::dal::svm::backend
