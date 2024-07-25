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

#ifdef ONEDAL_DATA_PARALLEL
#include <daal/include/data_management/data/internal/numeric_table_sycl_homogen.h>
#endif

#include "oneapi/dal/table/backend/interop/sycl_table_adapter.hpp"

namespace oneapi::dal::backend::interop {

#ifdef ONEDAL_DATA_PARALLEL
inline daal::data_management::NumericTablePtr convert_to_daal_table(const sycl::queue& queue,
                                                                    const table& table) {
    if (!table.has_data()) {
        return daal::data_management::NumericTablePtr{};
    }
    return interop::sycl_table_adapter::create(queue, table);
}

template <typename Data>
inline daal::data_management::NumericTablePtr convert_to_daal_table(const sycl::queue& queue,
                                                                    const array<Data>& data,
                                                                    std::int64_t row_count,
                                                                    std::int64_t column_count) {
    using daal::services::Status;
    using daal::services::SharedPtr;
    using daal::services::internal::Buffer;
    using daal::data_management::internal::SyclHomogenNumericTable;
    using dal::detail::integral_cast;

    ONEDAL_ASSERT(data.get_count() == row_count * column_count);
    ONEDAL_ASSERT(data.has_mutable_data());
    ONEDAL_ASSERT(is_same_context(queue, data));

    const SharedPtr<Data> data_shared{ data.get_mutable_data(), daal_object_owner{ data } };

    Status status;
    const Buffer<Data> buffer{ data_shared,
                               integral_cast<std::size_t>(data.get_count()),
                               queue,
                               status };
    status_to_exception(status);

    const auto table =
        SyclHomogenNumericTable<Data>::create(buffer,
                                              integral_cast<std::size_t>(column_count),
                                              integral_cast<std::size_t>(row_count),
                                              &status);
    status_to_exception(status);

    return table;
}
#endif

} // namespace oneapi::dal::backend::interop
