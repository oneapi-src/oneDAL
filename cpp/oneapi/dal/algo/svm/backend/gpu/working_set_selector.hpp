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

#ifdef ONEDAL_DATA_PARALLEL

enum class ws_edge { up, low };

inline std::int64_t propose_working_set_size(const sycl::queue& queue,
                                             const std::int64_t row_count) {
    const std::int64_t max_wg_size = dal::backend::device_max_wg_size(queue);
    return std::min(dal::backend::down_pow2<std::uint32_t>(row_count),
                    dal::backend::down_pow2<std::uint32_t>(max_wg_size));
}

template <typename Float>
class working_set_selector {
public:
    working_set_selector(const sycl::queue& queue,
                         const pr::ndarray<Float, 1>& labels,
                         const Float C,
                         const std::int64_t row_count);

    sycl::event select(const pr::ndview<Float, 1>& alpha,
                       const pr::ndview<Float, 1>& f,
                       pr::ndview<std::uint32_t, 1>& ws_indices,
                       const dal::backend::event_vector& deps = {});

private:
    sycl::event reset_indicator(const pr::ndview<std::uint32_t, 1>& idx,
                                pr::ndview<std::uint8_t, 1>& indicator,
                                const std::int64_t need_to_reset,
                                const dal::backend::event_vector& deps = {});

    std::tuple<const std::int64_t, sycl::event> select_ws_edge(
        const pr::ndview<Float, 1>& alpha,
        pr::ndview<std::uint32_t, 1>& ws_indices,
        const std::int64_t need_select_count,
        const std::int64_t left_to_select,
        ws_edge edge,
        const dal::backend::event_vector& deps = {});

    sycl::event check_ws_edge(sycl::queue& queue,
                              const pr::ndview<Float, 1>& y,
                              const pr::ndview<Float, 1>& alpha,
                              pr::ndview<std::uint8_t, 1>& indicator,
                              const Float C,
                              const std::int64_t n,
                              ws_edge edge,
                              const dal::backend::event_vector& deps = {});

    sycl::event sort_f_indices(sycl::queue& queue,
                               const pr::ndview<Float, 1>& f,
                               const dal::backend::event_vector& deps = {});

    sycl::queue queue_;

    std::int64_t row_count_;
    std::int64_t ws_count_;
    Float C_;

    pr::ndarray<std::uint32_t, 1> sorted_f_indices_;
    pr::ndarray<std::uint32_t, 1> buff_indices_;
    pr::ndarray<std::uint8_t, 1> indicator_;
    pr::ndarray<Float, 1> tmp_sort_values_;
    pr::ndarray<Float, 1> labels_;
};

#endif

} // namespace oneapi::dal::svm::backend
