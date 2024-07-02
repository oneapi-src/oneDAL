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

#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/table/backend/interop/sycl_table_adapter.hpp"
#include "oneapi/dal/table/backend/interop/host_homogen_table_adapter.hpp"
#include "oneapi/dal/table/backend/interop/host_soa_table_adapter.hpp"
#include "oneapi/dal/table/backend/interop/host_csr_table_adapter.hpp"
#include "oneapi/dal/backend/interop/csr_block_owner.hpp"

#ifdef ONEDAL_DATA_PARALLEL
#include "oneapi/dal/table/backend/interop/table_conversion_dpc.hpp"
#endif

namespace oneapi::dal::backend::interop {

namespace dm = daal::data_management;

using daal_csr_t = dm::CSRNumericTable;

using daal_soa_t = dm::SOANumericTable;

template <typename Data>
using daal_homogen_t = dm::HomogenNumericTable<Data>;

template <typename Data>
using daal_shared_t = daal::services::SharedPtr<Data>;

using numeric_table_ptr = dm::NumericTablePtr;

using csr_table_ptr = daal_shared_t<daal_csr_t>;

using soa_table_ptr = daal_shared_t<daal_soa_t>;
template <typename Data>
using homogen_table_ptr = daal_shared_t<daal_homogen_t<Data>>;

template <typename Data = DAAL_DATA_TYPE>
homogen_table_ptr<Data> allocate_daal_homogen_table(std::int64_t row_count,
                                                    std::int64_t column_count);

template <typename Data = DAAL_DATA_TYPE>
homogen_table_ptr<Data> empty_daal_homogen_table(std::int64_t column_count);

template <typename Data = DAAL_DATA_TYPE>
ONEDAL_EXPORT homogen_table_ptr<Data> convert_to_daal_homogen_table(array<Data>& data,
                                                                    std::int64_t row_count,
                                                                    std::int64_t column_count,
                                                                    bool allow_copy = false);
template <typename Data = DAAL_DATA_TYPE>
ONEDAL_EXPORT homogen_table_ptr<Data> copy_to_daal_homogen_table(const table& table);

template <typename Data = DAAL_DATA_TYPE>
ONEDAL_EXPORT homogen_table convert_from_daal_homogen_table(const numeric_table_ptr& nt);

ONEDAL_EXPORT numeric_table_ptr wrap_by_host_homogen_adapter(const homogen_table& table);

ONEDAL_EXPORT soa_table_ptr wrap_by_host_soa_adapter(const homogen_table& table);

ONEDAL_EXPORT soa_table_ptr wrap_by_host_soa_adapter(const heterogen_table& table);

template <typename Data = DAAL_DATA_TYPE>
ONEDAL_EXPORT numeric_table_ptr convert_to_daal_table(const homogen_table& table);

template <typename Data = DAAL_DATA_TYPE>
ONEDAL_EXPORT csr_table_ptr convert_to_daal_csr_table(array<Data>& data,
                                                      array<std::int64_t>& column_indices,
                                                      array<std::int64_t>& row_indices,
                                                      std::int64_t row_count,
                                                      std::int64_t column_count,
                                                      bool allow_copy = false);

template <typename Data = DAAL_DATA_TYPE>
ONEDAL_EXPORT csr_table_ptr copy_to_daal_csr_table(const csr_table& table);

template <typename Data = DAAL_DATA_TYPE>
ONEDAL_EXPORT csr_table convert_from_daal_csr_table(const numeric_table_ptr& nt);

ONEDAL_EXPORT csr_table_ptr wrap_by_host_csr_adapter(const csr_table& table);

ONEDAL_EXPORT soa_table_ptr wrap_by_host_soa_adapter(const heterogen_table& table);

template <typename Data = DAAL_DATA_TYPE>
ONEDAL_EXPORT numeric_table_ptr convert_to_daal_table(const heterogen_table& table);

ONEDAL_EXPORT heterogen_table convert_from_daal_heterogen_table(const numeric_table_ptr& nt);

template <typename Data = DAAL_DATA_TYPE>
ONEDAL_EXPORT csr_table_ptr convert_to_daal_table(const csr_table& table);

template <typename Data = DAAL_DATA_TYPE>
ONEDAL_EXPORT numeric_table_ptr convert_to_daal_table(const table& table);

template <typename Data = DAAL_DATA_TYPE>
ONEDAL_EXPORT table convert_from_daal_table(const numeric_table_ptr& nt);

} // namespace oneapi::dal::backend::interop
