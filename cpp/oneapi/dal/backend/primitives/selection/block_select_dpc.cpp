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

#include "oneapi/dal/backend/primitives/blas/block_select.hpp"
#include "oneapi/dal/backend/primitives/blas/single_pass_select.hpp"
#include "oneapi/dal/backend/primitives/blas/quick_select.hpp"

namespace oneapi::dal::backend::primitives {


template <typename Float, bool selected_out, bool indices_out>
sycl::event block_select(sycl::queue& queue,
                 ndview<Float, 2>& block,
                 std::int64_t k,
                 std::int64_t register_width,
                 ndview<Float, 2>& selected,
                 ndview<std::int64_t, 2>& indices,
                 const event_vector& deps) {
    ONEDAL_ASSERT(block.get_dimension(1) == selected.get_dimension(1));
    ONEDAL_ASSERT(block.get_dimension(1) == indices.get_dimension(1));
    ONEDAL_ASSERT(indices.get_dimension(0) == k);
    ONEDAL_ASSERT(selected.get_dimension(0) == k);
    ONEDAL_ASSERT(indices.has_mutable_data());
    ONEDAL_ASSERT(selection.has_mutable_data());

//    if (k <= register_width) {
        return single_pass_select<Float, selected_out, indices_out>(queue,
                               block,
                               k,
                               selected,
                               indices,
                               deps);
//    }
//    else {
//        return quick_select<Float, selected_out, indices_out>(queue,
//                               block,
//                               selected,
//                               indices,
//                               k,
//                               deps);
//    }
}

#define INSTANTIATE(F, selected_out, indices_out)                                                    \
    template ONEDAL_EXPORT sycl::event block_select<F, sel_out, ind_out>(sycl::queue & queue,       \
                 ndview<Float, 2>& block, \
                 std::int64_t k, \
                 std::int64_t register_width, \
                 ndview<Float, 2>& selected, \
                 ndview<std::int64_t, 2>& indices, \
                 const event_vector& deps);

#define INSTANTIATE_FLOAT(selected_out, indices_out) \
    INSTANTIATE(float, selected_out, indices_out)    \
    INSTANTIATE(double, selected_out, indices_out)

INSTANTIATE_FLOAT(true, false)
INSTANTIATE_FLOAT(false, true)
INSTANTIATE_FLOAT(true, true)

} // namespace oneapi::dal::backend::primitives
