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

template <typename TypeND>
auto copy_by_indices(const sycl::queue& q,
                     const TypeND& x,
                     const pr::ndview<std::uint32_t, 1>& x_indices,
                     TypeND& res,
                     const std::int64_t row_count,
                     const std::int64_t column_count,
                     const dal::backend::event_vector& deps = {}) {
    ONEDAL_ASSERT(x_indices.get_count()) == row_count;
    ONEDAL_ASSERT(res.get_count() == row_count * column_count);
    ONEDAL_ASSERT(res.has_mutable_data());

    const Float* x_ptr = x.get_data();
    const std::uint32_t* x_indices_ptr = x_indices.get_mutable_data();
    Float* res_ptr = res.get_mutable_data();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = dal::backend::make_multiple_nd_range_1d(column_count, row_count);
        cgh.depends_on(deps);

        cgh.parallel_for(range, [=](sycl::nd_item<1> item) {
            const std::uint32_t row_index = item.get_global_id(1);
            const std::uint32_t col_index = item.get_global_id(0);

            const std::uint32_t row_x_index = x_indices_ptr[row_index];

            const Float* const x_i = &x_ptr[row_x_index * column_count];
            Float* res_i = &res_ptr[row_index * column_count];
            res_i[col_index] = x_i[col_index];
        });
    });
}

inline sycl::event invert_values(sycl::queue& q,
                                 const pr::ndview<Float, 1>& data,
                                 pr::ndview<Float, 1>& res,
                                 const dal::backend::event_vector& deps = {}) {
    ONEDAL_ASSERT(data.get_dimension(0) == res.get_dimension(0));

    const Float* data_ptr = data.get_data();
    Float* res_ptr = res.get_mutable_data();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = dal::backend::make_range_1d(data.get_count());
        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            res_ptr[idx] = -data_ptr[idx];
        });
    });
}

template <typename Type>
std::tuple<const std::int64_t, sycl::event> copy_last_to_first(
    const sycl::queue& q,
    pr::ndview<Type, 1>& data,
    const dal::backend::event_vector& deps = {}) {
    ONEDAL_ASSERT(data.has_mutable_data());
    std::int64_t data_size = data.get_count();
    const std::int64_t copy_count = data_size / 2;
    Type* data_ptr = data.get_mutable_data();
    auto copy_event =
        dal::backend::copy(q, data_ptr, data_ptr + copy_count, data_size - copy_count, deps);
    return { copy_count, copy_event };
}

} // namespace oneapi::dal::svm::backend
