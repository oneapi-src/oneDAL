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
#include "oneapi/dal/detail/profiler.hpp"

namespace oneapi::dal::svm::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace pr = dal::backend::primitives;

enum class violating_edge { up, low };

template <typename Float>
inline bool is_upper_edge(const Float& y, const Float& alpha, const Float& C) {
    return (y > 0 && alpha < C) || (y < 0 && alpha > 0);
}

template <typename Float>
inline bool is_lower_edge(const Float& y, const Float& alpha, const Float& C) {
    return (y > 0 && alpha > 0) || (y < 0 && alpha < C);
}

template <typename Float>
sycl::event check_violating_edge(sycl::queue& q,
                                 const pr::ndview<Float, 1>& y,
                                 const pr::ndview<Float, 1>& alpha,
                                 pr::ndview<std::uint8_t, 1>& indicator,
                                 const Float C,
                                 const violating_edge edge,
                                 const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(check_violating_edge, q);
    ONEDAL_ASSERT(y.get_dimension(0) == alpha.get_dimension(0));
    ONEDAL_ASSERT(alpha.get_dimension(0) == indicator.get_dimension(0));
    ONEDAL_ASSERT(indicator.has_mutable_data());

    const Float* y_ptr = y.get_data();
    const Float* alpha_ptr = alpha.get_data();
    std::uint8_t* indicator_ptr = indicator.get_mutable_data();

    const std::int32_t row_count = y.get_dimension(0);

    const auto wg_size =
        std::min(dal::detail::integral_cast<std::int32_t>(dal::backend::propose_wg_size(q)),
                 row_count);
    const auto range = dal::backend::make_multiple_nd_range_1d(row_count, wg_size);

    sycl::event check_event;

    if (edge == violating_edge::up) {
        check_event = q.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);

            cgh.parallel_for(range, [=](sycl::nd_item<1> item) {
                const std::int32_t i = item.get_global_id(0);
                indicator_ptr[i] =
                    static_cast<std::uint8_t>(is_upper_edge<Float>(y_ptr[i], alpha_ptr[i], C));
            });
        });
    }
    else {
        check_event = q.submit([&](sycl::handler& cgh) {
            cgh.depends_on(deps);

            cgh.parallel_for(range, [=](sycl::nd_item<1> item) {
                const std::int32_t i = item.get_global_id(0);
                indicator_ptr[i] =
                    static_cast<std::uint8_t>(is_lower_edge<Float>(y_ptr[i], alpha_ptr[i], C));
            });
        });
    }
    return check_event;
}

template <typename Float, typename Integer>
auto copy_by_indices(sycl::queue& q,
                     const pr::ndview<Float, 2>& x,
                     const pr::ndview<Integer, 1>& x_indices,
                     pr::ndview<Float, 2>& res,
                     const std::int32_t row_count,
                     const std::int32_t column_count,
                     const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(copy_by_indices, q);
    ONEDAL_ASSERT(x_indices.get_count() == row_count);
    ONEDAL_ASSERT(res.get_count() == row_count * column_count);
    ONEDAL_ASSERT(x.get_dimension(1) == column_count);
    ONEDAL_ASSERT(column_count > 0);
    ONEDAL_ASSERT(row_count > 0);
    ONEDAL_ASSERT(res.has_mutable_data());

    const Float* x_ptr = x.get_data();
    const Integer* x_indices_ptr = x_indices.get_mutable_data();
    Float* res_ptr = res.get_mutable_data();

    return q.submit([&](sycl::handler& cgh) {
        const auto range = dal::backend::make_range_2d(column_count, row_count);

        cgh.depends_on(deps);
        cgh.parallel_for(range, [=](sycl::id<2> idx) {
            const auto& col_index = idx[0];
            const auto& row_index = idx[1];

            const Integer row_x_index = x_indices_ptr[row_index];

            const Float* const x_i = &x_ptr[row_x_index * column_count];
            Float* res_i = &res_ptr[row_index * column_count];
            res_i[col_index] = x_i[col_index];
        });
    });
}

template <typename Float>
inline sycl::event invert_values(sycl::queue& q,
                                 const pr::ndview<Float, 1>& data,
                                 pr::ndview<Float, 1>& res,
                                 const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(invert_values, q);
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(res.has_mutable_data());
    ONEDAL_ASSERT(data.get_count() == res.get_count());

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
std::tuple<const std::int32_t, sycl::event> copy_last_to_first(
    sycl::queue& q,
    pr::ndview<Type, 1>& data,
    const dal::backend::event_vector& deps = {}) {
    ONEDAL_PROFILER_TASK(copy_last_to_first, q);
    ONEDAL_ASSERT(data.has_mutable_data());
    std::int32_t data_size = data.get_count();
    const std::int32_t copy_count = data_size / 2;
    Type* data_ptr = data.get_mutable_data();
    auto copy_event =
        dal::backend::copy(q, data_ptr, data_ptr + copy_count, data_size - copy_count, deps);
    return { copy_count, copy_event };
}

#endif

} // namespace oneapi::dal::svm::backend
