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

#include "oneapi/dal/backend/primitives/selection/block_quick_select.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float, bool selection_out, bool indices_out>
sycl::event block_quick_select(sycl::queue& queue,
                               const ndview<Float, 2>& block,
                               std::int64_t k,
                               ndview<Float, 2>& selection,
                               ndview<int, 2>& indices,
                               const event_vector& deps) {
    ONEDAL_ASSERT(block.get_dimension(1) == selection.get_dimension(1));
    ONEDAL_ASSERT(block.get_dimension(1) == indices.get_dimension(1));
    ONEDAL_ASSERT(indices.get_dimension(0) == k);
    ONEDAL_ASSERT(selection.get_dimension(0) == k);
    ONEDAL_ASSERT(indices.has_mutable_data());
    ONEDAL_ASSERT(selection.has_mutable_data());

    return sycl::event();
}

#define INSTANTIATE(F, selection_out, indices_out)                                        \
    template ONEDAL_EXPORT sycl::event block_quick_select<F, selection_out, indices_out>( \
        sycl::queue & queue,                                                              \
        const ndview<F, 2>& block,                                                        \
        std::int64_t k,                                                                   \
        ndview<F, 2>& selection,                                                          \
        ndview<int, 2>& indices,                                                          \
        const event_vector& deps);

#define INSTANTIATE_FLOAT(selection_out, indices_out) \
    INSTANTIATE(float, selection_out, indices_out)    \
    INSTANTIATE(double, selection_out, indices_out)

INSTANTIATE_FLOAT(true, false)
INSTANTIATE_FLOAT(false, true)
INSTANTIATE_FLOAT(true, true)

} // namespace oneapi::dal::backend::primitives
