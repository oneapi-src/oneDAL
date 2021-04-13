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

inline sycl::event make_range(sycl::queue& queue,
                              pr::ndarray<std::uint32_t, 1>& indices_sort,
                              const std::uint32_t n,
                              const dal::backend::event_vector& deps = {}) {
    ONEDAL_ASSERT(indices_sort.get_dimension(0) == n);
    ONEDAL_ASSERT(indices_sort.has_mutable_data());

    std::uint32_t* indices_sort_ptr = indices_sort.get_mutable_data();

    auto make_range_event = queue.submit([&](sycl::handler& chg) {
        chg.depends_on(deps);
        const auto range_dim = dal::detail::integral_cast<std::size_t>(n);
        const auto range = sycl::range<1>(range_dim);

        chg.parallel_for(range, [=](sycl::id<1> id) {
            const uint i = id[0];
            indices_sort_ptr[i] = i;
        });
    });

    return make_range_event;
}

template <typename Float>
inline sycl::event arg_sort(sycl::queue& queue,
                            const pr::ndarray<Float, 1>& f,
                            pr::ndarray<Float, 1>& values,
                            pr::ndarray<std::uint32_t, 1>& indices_sort,
                            const std::uint32_t n,
                            const dal::backend::event_vector& deps = {}) {
    ONEDAL_ASSERT(f.get_dimension(0) == n);
    ONEDAL_ASSERT(values.get_dimension(0) == n);
    ONEDAL_ASSERT(values.has_mutable_data());
    sycl::event::wait_and_throw(deps);

    Float* f_ptr = f.get_data();
    Float* values_ptr = values.get_mutable_data();

    auto copy_event = dal::backend::copy(queue, values_ptr, f_ptr, n);
    auto make_range_event = make_range(queue, indices_sort, n);
    auto radix_sort_event = pr::radix_sort_indices_inplace<Float, std::uint32_t>{
        queue
    }(values, indices_sort, { copy_event, make_range_event });

    return radix_sort_event;
}

template <typename Float>
inline sycl::event check_upper(sycl::queue& queue,
                               const pr::ndarray<Float, 1>& y,
                               const pr::ndarray<Float, 1>& alpha,
                               pr::ndarray<std::uint32_t, 1>& indicator,
                               const Float C,
                               const std::uint32_t n,
                               const dal::backend::event_vector& deps = {}) {
    ONEDAL_ASSERT(y.get_dimension(0) == n);
    ONEDAL_ASSERT(alpha.get_dimension(0) == n);
    ONEDAL_ASSERT(indicator.get_dimension(0) == n);
    ONEDAL_ASSERT(indicator.has_mutable_data());

    Float* y_ptr = y.get_data();
    Float* alpha_ptr = alpha.get_data();
    Float* indicator_ptr = indicator.get_mutable_data();

    auto check_upper_event = queue.submit([&](sycl::handler& chg) {
        chg.depends_on(deps);
        const auto range_dim = dal::detail::integral_cast<std::size_t>(n);
        const auto range = sycl::range<1>(range_dim);

        chg.parallel_for(range, [=](sycl::id<1> id) {
            const uint i = id[0];
            indicator_ptr[i] =
                (y_ptr[i] > 0 && alpha_ptr[i] < C) || (y_ptr[i] < 0 && alpha_ptr[i] > 0);
        });
    });

    return check_upper_event;
}

template <typename Float>
inline sycl::event check_lower(sycl::queue& queue,
                               const pr::ndarray<Float, 1>& y,
                               const pr::ndarray<Float, 1>& alpha,
                               pr::ndarray<std::uint32_t, 1>& indicator,
                               const Float C,
                               const std::uint32_t n,
                               const dal::backend::event_vector& deps = {}) {
    ONEDAL_ASSERT(y.get_dimension(0) == n);
    ONEDAL_ASSERT(alpha.get_dimension(0) == n);
    ONEDAL_ASSERT(indicator.get_dimension(0) == n);
    ONEDAL_ASSERT(indicator.has_mutable_data());

    Float* y_ptr = y.get_data();
    Float* alpha_ptr = alpha.get_data();
    Float* indicator_ptr = indicator.get_mutable_data();

    auto check_upper_event = queue.submit([&](sycl::handler& chg) {
        chg.depends_on(deps);
        const auto range_dim = dal::detail::integral_cast<std::size_t>(n);
        const auto range = sycl::range<1>(range_dim);

        chg.parallel_for(range, [=](sycl::id<1> id) {
            const uint i = id[0];
            indicator_ptr[i] =
                (y_ptr[i] > 0 && alpha_ptr[i] > 0) || (y_ptr[i] < 0 && alpha_ptr[i] < C);
        });
    });

    return check_upper_event;
}

} // namespace oneapi::dal::svm::backend
