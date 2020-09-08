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

#ifdef ONEAPI_DAL_DATA_PARALLEL
#define DAAL_SYCL_INTERFACE
#define DAAL_SYCL_INTERFACE_USM
#include <daal/include/data_management/data/numeric_table_sycl_homogen.h>
#endif
#include <daal/include/data_management/data/homogen_numeric_table.h>

#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::backend::interop {

template <typename T>
struct daal_array_owner {
    explicit daal_array_owner(const array<T>& arr) : array_(arr) {}

    void operator()(const void*) {
        array_.reset();
    }

    array<T> array_;
};

template <typename T>
inline auto allocate_daal_homogen_table(std::int64_t row_count, std::int64_t column_count) {
    return daal::data_management::HomogenNumericTable<T>::create(
        column_count,
        row_count,
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
        daal::services::SharedPtr<T>(data.get_mutable_data(), daal_array_owner<T>{ data });

    return daal::data_management::HomogenNumericTable<T>::create(daal_data,
                                                                 column_count,
                                                                 row_count);
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

#ifdef ONEAPI_DAL_DATA_PARALLEL
template <typename T>
inline auto convert_to_daal_sycl_homogen_table(sycl::queue& queue,
                                               array<T>& data,
                                               std::int64_t row_count,
                                               std::int64_t column_count) {
    data.need_mutable_data(queue);
    const auto daal_data =
        daal::services::SharedPtr<T>(data.get_mutable_data(), daal_array_owner<T>{ data });
    return daal::data_management::SyclHomogenNumericTable<T>::create(daal_data,
                                                                     column_count,
                                                                     row_count,
                                                                     cl::sycl::usm::alloc::shared);
}

#endif

} // namespace oneapi::dal::backend::interop
