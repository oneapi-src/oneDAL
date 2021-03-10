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

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

/// Performs K-selection on each row in matrix
///
/// @param[in]  queue The queue
/// @param[in]  block  The [n x m] input matrix (m >= k)
/// @param[in]  k      The number of minimal values to be selected in each row
/// @param[in] register_width  The maximal possible size of array in private memory
/// @param[out] selected The [n x k] matrix of selected values (if selected_out == true)
/// @param[out] indices  The [n x k] matrix of indices of selected values (if indices_out == true)

template <typename Float, bool selected_out, bool indices_out>
sycl::event block_select(sycl::queue& queue,
                         ndview<Float, 2>& block,
                         std::int64_t k,
                         ndview<Float, 2>& selected,
                         ndview<int, 2>& indices,
                         const event_vector& deps = {});

#endif

} // namespace oneapi::dal::backend::primitives
