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

#include <limits>

#include "oneapi/dal/backend/primitives/selection/select_by_rows_quick.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float, bool selection_out, bool indices_out>
sycl::event select_by_rows_quick(sycl::queue& queue,
                                 const ndview<Float, 2>& data,
                                 std::int64_t k,
                                 std::int64_t col_begin,
                                 std::int64_t col_end,
                                 ndview<Float, 2>& selection,
                                 ndview<int, 2>& indices,
                                 const event_vector& deps) {
    ONEDAL_ASSERT(false);
    return sycl::event();
}

#define INSTANTIATE(F, selection_out, indices_out)                                          \
    template ONEDAL_EXPORT sycl::event select_by_rows_quick<F, selection_out, indices_out>( \
        sycl::queue & queue,                                                                \
        const ndview<F, 2>& block,                                                          \
        std::int64_t k,                                                                     \
        std::int64_t col_begin,                                                             \
        std::int64_t col_end,                                                               \
        ndview<F, 2>& selection,                                                            \
        ndview<int, 2>& indices,                                                            \
        const event_vector& deps);

#define INSTANTIATE_FLOAT(selection_out, indices_out) \
    INSTANTIATE(float, selection_out, indices_out)    \
    INSTANTIATE(double, selection_out, indices_out)

INSTANTIATE_FLOAT(true, false)
INSTANTIATE_FLOAT(false, true)
INSTANTIATE_FLOAT(true, true)

} // namespace oneapi::dal::backend::primitives
