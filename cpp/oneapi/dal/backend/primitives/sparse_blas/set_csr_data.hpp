/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/table/common.hpp"

#ifdef ONEDAL_DATA_PARALLEL
#include <oneapi/mkl.hpp>
#endif // ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

template <typename Float>
sycl::event set_csr_data(sycl::queue& queue,
                         oneapi::mkl::sparse::matrix_handle_t handle,
                         const std::int64_t column_count,
                         dal::sparse_indexing indexing,
                         dal::array<Float> &data,
                         dal::array<std::int64_t> &column_indices,
                         dal::array<std::int64_t> &row_offsets,
                         const event_vector& deps = {});

#endif // ifdef ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::backend::primitives

