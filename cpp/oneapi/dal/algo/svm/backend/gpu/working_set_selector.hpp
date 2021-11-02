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
#include "oneapi/dal/algo/svm/backend/gpu/misc.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::svm::backend {

namespace pr = dal::backend::primitives;

inline std::int32_t propose_working_set_size(const sycl::queue& q, const std::int32_t row_count) {
    const std::int32_t max_wg_size = dal::backend::device_max_wg_size(q);
    return std::min(dal::backend::down_pow2<std::int32_t>(row_count),
                    dal::backend::down_pow2<std::int32_t>(max_wg_size));
}

template <typename Float>
class working_set_selector {
public:
    working_set_selector(const sycl::queue& q,
                         const pr::ndarray<Float, 1>& labels,
                         const Float C,
                         const std::int32_t row_count);

    sycl::event select(const pr::ndview<Float, 1>& alpha,
                       const pr::ndview<Float, 1>& f,
                       pr::ndview<std::int32_t, 1>& ws_indices,
                       std::int32_t selected_count,
                       const dal::backend::event_vector& deps = {});

private:
    sycl::event reset_indicator(const pr::ndview<std::int32_t, 1>& idx,
                                pr::ndview<std::uint8_t, 1>& indicator,
                                const std::int32_t need_to_reset,
                                const dal::backend::event_vector& deps = {});

    std::tuple<const std::int32_t, sycl::event> select_ws_edge(
        const pr::ndview<Float, 1>& alpha,
        pr::ndview<std::int32_t, 1>& ws_indices,
        const std::int32_t need_select_count,
        const std::int32_t already_selected,
        violating_edge edge,
        const dal::backend::event_vector& deps = {});

    sycl::event sort_f_indices(sycl::queue& q,
                               const pr::ndview<Float, 1>& f,
                               const dal::backend::event_vector& deps = {});

    sycl::queue q_;

    std::int32_t row_count_;
    std::int32_t ws_count_;
    Float C_;

    pr::ndarray<std::int32_t, 1> sorted_f_indices_;
    pr::ndarray<std::int32_t, 1> buff_indices_;
    pr::ndarray<std::uint8_t, 1> indicator_;
    pr::ndarray<Float, 1> tmp_sort_values_;
    pr::ndarray<Float, 1> labels_;
};

} // namespace oneapi::dal::svm::backend

#endif
