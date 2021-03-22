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

#include "oneapi/dal/backend/primitives/selection/row_partitioning.hpp"

namespace oneapi::dal::backend::primitives {

#ifdef ONEDAL_DATA_PARALLEL

sycl::nd_range<2> get_row_partitioning_range(std::int64_t row_count, std::int64_t col_count);

template<typename Float>
int SYCL_EXTERNAL kernel_row_partitioning(const sycl::stream& out, sycl::nd_item<2> item,
                                           Float* values,
                                           int* indices,
                                           int partition_start,
                                           int partition_end,
                                           Float pivot);

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::backend::primitives
