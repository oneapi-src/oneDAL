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

#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/table/backend/interop/host_homogen_table_adapter.hpp"

namespace oneapi::dal::backend::interop {

template <typename T>
inline auto allocate_daal_homogen_table(std::int64_t row_count, std::int64_t column_count) {
    return daal::data_management::HomogenNumericTable<T>::create(
        dal::detail::integral_cast<std::size_t>(column_count),
        dal::detail::integral_cast<std::size_t>(row_count),
        daal::data_management::NumericTable::doAllocate);
}

template <typename T>
inline auto convert_to_daal_homogen_table(array<T>& data,
                                          std::int64_t row_count,
                                          std::int64_t column_count) {
    if (!data.get_count())
        return daal::services::SharedPtr<daal::data_management::HomogenNumericTable<T>>();
    data.need_mutable_data();
    const auto daal_data =
        daal::services::SharedPtr<T>(data.get_mutable_data(), daal_object_owner{ data });

    return daal::data_management::HomogenNumericTable<T>::create(
        daal_data,
        dal::detail::integral_cast<std::size_t>(column_count),
        dal::detail::integral_cast<std::size_t>(row_count));
}

template <typename T>
inline table convert_from_daal_homogen_table(const daal::data_management::NumericTablePtr& nt) {
    daal::data_management::BlockDescriptor<T> block;
    const std::int64_t row_count = nt->getNumberOfRows();
    const std::int64_t column_count = nt->getNumberOfColumns();

    nt->getBlockOfRows(0, row_count, daal::data_management::readOnly, block);
    T* data = block.getBlockPtr();
    array<T> arr(data, row_count * column_count, [nt, block](T* p) mutable {
        nt->releaseBlockOfRows(block);
    });
    return detail::homogen_table_builder{}.reset(arr, row_count, column_count).build();
}

inline NumericTablePtr wrap_by_host_homogen_adapter(const homogen_table& table) {
    const auto& dtype = table.get_metadata().get_data_type(0);

    switch (dtype) {
        case data_type::float32: return host_homogen_table_adapter<float>::create(table);
        case data_type::float64: return host_homogen_table_adapter<double>::create(table);
        case data_type::int32: return host_homogen_table_adapter<std::int32_t>::create(table);
        default: return NumericTablePtr();
    }
}

template <typename Float>
inline NumericTablePtr copy_to_daal_homogen_table(const table& table) {
    auto rows = row_accessor<const Float>{ table }.pull();
    return convert_to_daal_homogen_table(rows,
                                         table.get_row_count(),
                                         table.get_column_count());
}

template <typename Float>
inline NumericTablePtr convert_to_daal_table(const homogen_table& table) {
    auto wrapper = wrap_by_host_homogen_adapter(table);
    if (!wrapper) {
        return copy_to_daal_homogen_table<Float>(table);
    }
    else {
        return wrapper;
    }
}

template <typename Float>
inline NumericTablePtr convert_to_daal_table(const table& table) {
    if (table.get_kind() == homogen_table::kind()) {
        const auto& homogen = static_cast<const homogen_table&>(table);
        return convert_to_daal_table<Float>(homogen);
    }
    else {
        return copy_to_daal_homogen_table<Float>(table);
    }
}


#ifdef ONEDAL_DATA_PARALLEL
template <typename T>
inline auto convert_to_daal_sycl_homogen_table(sycl::queue& queue,
                                               array<T>& data,
                                               std::int64_t row_count,
                                               std::int64_t column_count) {
    data.need_mutable_data(queue);
    const auto daal_data =
        daal::services::SharedPtr<T>(data.get_mutable_data(), daal_object_owner{ data });

    using daal::data_management::internal::SyclHomogenNumericTable;
    return SyclHomogenNumericTable<T>::create(daal_data,
                                              dal::detail::integral_cast<std::size_t>(column_count),
                                              dal::detail::integral_cast<std::size_t>(row_count),
                                              cl::sycl::usm::alloc::shared);
}

#endif

} // namespace oneapi::dal::backend::interop
