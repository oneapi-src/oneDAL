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

#include <type_traits>

#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float>
class kselect_by_rows_base {
public:
    virtual ~kselect_by_rows_base() {}

    virtual sycl::event operator()(sycl::queue& queue,
                                   const ndview<Float, 2>& data,
                                   std::int64_t k,
                                   ndview<Float, 2>& selection,
                                   ndview<std::int32_t, 2>& column_indices,
                                   const event_vector& deps = {}) = 0;

    virtual sycl::event operator()(sycl::queue& queue,
                                   const ndview<Float, 2>& data,
                                   std::int64_t k,
                                   ndview<Float, 2>& selection,
                                   const event_vector& deps = {}) = 0;

    virtual sycl::event operator()(sycl::queue& queue,
                                   const ndview<Float, 2>& data,
                                   std::int64_t k,
                                   ndview<std::int32_t, 2>& column_indices,
                                   const event_vector& deps = {}) = 0;

    virtual sycl::event select_sq_l2(sycl::queue& queue,
                                     const ndview<Float, 1>& n1,
                                     const ndview<Float, 1>& n2,
                                     const ndview<Float, 2>& ip,
                                     std::int64_t k,
                                     ndview<Float, 2>& selection,
                                     ndview<std::int32_t, 2>& column_indices,
                                     const event_vector& deps = {}) = 0;

    virtual sycl::event select_sq_l2(sycl::queue& queue,
                                     const ndview<Float, 1>& n1,
                                     const ndview<Float, 1>& n2,
                                     const ndview<Float, 2>& ip,
                                     std::int64_t k,
                                     ndview<Float, 2>& selection,
                                     const event_vector& deps = {}) = 0;

    virtual sycl::event select_sq_l2(sycl::queue& queue,
                                     const ndview<Float, 1>& n1,
                                     const ndview<Float, 1>& n2,
                                     const ndview<Float, 2>& ip,
                                     std::int64_t k,
                                     ndview<std::int32_t, 2>& column_indices,
                                     const event_vector& deps = {}) = 0;
};

inline std::int64_t get_scaled_wg_size_per_row(const sycl::queue& queue,
                                               std::int64_t col_count,
                                               std::int64_t preffered_wg_size) {
    const std::int64_t sg_max_size = device_max_sg_size(queue);
    const std::int64_t row_adjusted_sg_num =
        col_count / sg_max_size + std::int64_t(col_count % sg_max_size > 0);
    std::int64_t expected_sg_num = std::min(preffered_wg_size / sg_max_size, row_adjusted_sg_num);
    if (expected_sg_num < 1)
        expected_sg_num = 1;
    return dal::detail::check_mul_overflow(expected_sg_num, sg_max_size);
}

#endif

} // namespace oneapi::dal::backend::primitives
