/*******************************************************************************
* Copyright 2020 Intel Corporation
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
#include <daal/include/data_management/data/numeric_table.h>

namespace oneapi::dal::backend::interop {

template <typename Data>
daal::data_management::NumericTablePtr allocate_daal_homogen_table(std::int64_t row_count,
                                                                   std::int64_t column_count);

template <typename Data>
daal::data_management::NumericTablePtr convert_to_daal_homogen_table(array<Data>& data,
                                                                     std::int64_t row_count,
                                                                     std::int64_t column_count);

template <typename Float>
daal::data_management::NumericTablePtr convert_to_daal_table(const table& table);

template <typename Data>
table convert_from_daal_homogen_table(const daal::data_management::NumericTablePtr& nt);

#ifdef ONEAPI_DAL_DATA_PARALLEL
template <typename Data>
daal::data_management::NumericTablePtr convert_to_daal_sycl_homogen_table(
    sycl::queue& queue,
    array<Data>& data,
    std::int64_t row_count,
    std::int64_t column_count);

template <typename Float>
daal::data_management::NumericTablePtr convert_to_daal_table(
    const detail::data_parallel_policy& policy,
    const table& table);

#endif

} // namespace oneapi::dal::backend::interop
