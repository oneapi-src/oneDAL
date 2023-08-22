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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/chunked_array.hpp"

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/backend/common_kernels.hpp"

namespace oneapi::dal::backend {

using heterogen_data = array<detail::chunked_array_base>;

template <typename Policy, typename Type>
void heterogen_pull_rows(const Policy& policy,
                         const table_metadata& meta,
                         const heterogen_data& data,
                         array<Type>& block_data,
                         const range& rows_range,
                         alloc_kind requested_alloc_kind);

template <typename Policy, typename Type>
void heterogen_pull_column(const Policy& policy,
                           const table_metadata& meta,
                           const heterogen_data& data,
                           array<Type>& block_data,
                           std::int64_t column,
                           const range& rows_range,
                           alloc_kind requested_alloc_kind);

heterogen_data heterogen_row_slice(const range& rows_range,
                                   const table_metadata& meta,
                                   const heterogen_data& data);

std::int64_t heterogen_column_count(const table_metadata& meta, const heterogen_data& data);

std::int64_t heterogen_row_count(std::int64_t column_count,
                                 const table_metadata& meta,
                                 const heterogen_data& data);

std::int64_t heterogen_row_count(const table_metadata& meta, const heterogen_data& data);

std::pair<std::int64_t, std::int64_t> heterogen_shape(const table_metadata& meta,
                                                      const heterogen_data& data);

} // namespace oneapi::dal::backend
