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

#include "oneapi/dal/table/backend/interop/conversion.hpp"

#ifdef ONEAPI_DAL_DATA_PARALLEL
#include <daal/include/data_management/data/internal/numeric_table_sycl_homogen.h>
#endif
#include <daal/include/data_management/data/homogen_numeric_table.h>

#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/table/backend/interop/homogen_table_adapter.hpp"

namespace oneapi::dal::backend::interop {

using daal::data_management::NumericTablePtr;
using std::int64_t;

template <typename Data>
NumericTablePtr allocate_daal_homogen_table(int64_t row_count, int64_t column_count) {
    using namespace daal::data_management;

    return HomogenNumericTable<Data>::create(column_count, row_count, NumericTable::doAllocate);
}

template <typename Data>
NumericTablePtr convert_to_daal_homogen_table(array<Data>& data,
                                              int64_t row_count,
                                              int64_t column_count) {
    using namespace daal::data_management;
    if (!data.get_count())
        return daal::services::SharedPtr<HomogenNumericTable<Data>>();
    data.need_mutable_data();
    const auto daal_data =
        daal::services::SharedPtr<Data>(data.get_mutable_data(), daal_array_owner{ data });

    return HomogenNumericTable<Data>::create(daal_data, column_count, row_count);
}

template <typename Data, typename Policy = detail::default_host_policy>
NumericTablePtr create_homogen_adapter(const Policy& policy, const homogen_table& origin) {
    return homogen_table_adapter<Policy, Data>::create(policy, origin);
}

template <typename Policy>
NumericTablePtr wrap_by_homogen_adapter(const Policy& policy, const homogen_table& table) {
    const auto& dtype = table.get_metadata().get_data_type(0);

    switch (dtype) {
        case data_type::float32: return create_homogen_adapter<float>(policy, table);
        case data_type::float64: return create_homogen_adapter<double>(policy, table);
        case data_type::int32: return create_homogen_adapter<std::int32_t>(policy, table);
        default: return NumericTablePtr();
    }
}

template <typename Float>
NumericTablePtr convert_to_daal_table(const table& table) {
    if (table.get_kind() == homogen_table::kind()) {
        const auto& homogen = static_cast<const homogen_table&>(table);
        auto wrapper = wrap_by_homogen_adapter(detail::default_host_policy{}, homogen);
        if (!wrapper) {
            auto rows = row_accessor<const Float>{ homogen }.pull();
            return convert_to_daal_homogen_table(rows,
                                                 homogen.get_row_count(),
                                                 homogen.get_column_count());
        }
        else {
            return wrapper;
        }
    }
    else {
        auto rows = row_accessor<const Float>{ table }.pull();
        return convert_to_daal_homogen_table(rows, table.get_row_count(), table.get_column_count());
    }
}

template <typename Data>
table convert_from_daal_homogen_table(const NumericTablePtr& nt) {
    daal::data_management::BlockDescriptor<Data> block;
    const std::int64_t row_count = nt->getNumberOfRows();
    const std::int64_t column_count = nt->getNumberOfColumns();

    nt->getBlockOfRows(0, row_count, daal::data_management::readOnly, block);
    Data* data = block.getBlockPtr();
    array<Data> arr(data, row_count * column_count, [nt, block](Data* p) mutable {
        nt->releaseBlockOfRows(block);
    });
    return detail::homogen_table_builder{}.reset(arr, row_count, column_count).build();
}

#ifdef ONEAPI_DAL_DATA_PARALLEL
template <typename Data>
NumericTablePtr convert_to_daal_sycl_homogen_table(sycl::queue& queue,
                                                   array<Data>& data,
                                                   int64_t row_count,
                                                   int64_t column_count) {
    data.need_mutable_data(queue);
    const auto daal_data =
        daal::services::SharedPtr<Data>(data.get_mutable_data(), daal_array_owner<Data>{ data });
    return daal::data_management::internal::SyclHomogenNumericTable<Data>::create(
        daal_data,
        column_count,
        row_count,
        cl::sycl::usm::alloc::shared);
}

template <typename Float>
NumericTablePtr convert_to_daal_table(const detail::data_parallel_policy& policy,
                                      const table& table) {
    if (table.get_kind() == homogen_table::kind()) {
        const auto& homogen = static_cast<const homogen_table&>(table);
        auto wrapper = wrap_by_homogen_adapter(policy, homogen);
        if (!wrapper) {
            auto& queue = policy.get_queue();
            auto rows = row_accessor<const Float>{ homogen }.pull(queue);
            return convert_to_daal_sycl_homogen_table(queue,
                                                      rows,
                                                      homogen.get_row_count(),
                                                      homogen.get_column_count());
        }
        else {
            return wrapper;
        }
    }
    else {
        auto& queue = policy.get_queue();
        auto rows = row_accessor<const Float>{ table }.pull(queue);
        return convert_to_daal_sycl_homogen_table(queue,
                                                  rows,
                                                  table.get_row_count(),
                                                  table.get_column_count());
    }
}

#endif

#define INSTANTIATE_ALL_HOST_METHODS(Data)                                                  \
    template NumericTablePtr allocate_daal_homogen_table<Data>(int64_t, int64_t);           \
    template NumericTablePtr convert_to_daal_homogen_table(array<Data>&, int64_t, int64_t); \
    template NumericTablePtr convert_to_daal_table<Data>(const table& table);               \
    template table convert_from_daal_homogen_table<Data>(const NumericTablePtr& nt);

#ifdef ONEAPI_DAL_DATA_PARALLEL

#define INSTANTIATE_ALL_METHODS(Data)                                                         \
    INSTANTIATE_ALL_HOST_METHODS(Data)                                                        \
    template NumericTablePtr convert_to_daal_sycl_homogen_table(sycl::queue&,                 \
                                                                array<Data>&,                 \
                                                                int64_t,                      \
                                                                int64_t);                     \
    template NumericTablePtr convert_to_daal_table<Data>(const detail::data_parallel_policy&, \
                                                         const table&);
#else

#define INSTANTIATE_ALL_METHODS(Data) INSTANTIATE_ALL_HOST_METHODS(Data)
#endif

INSTANTIATE_ALL_METHODS(float)
INSTANTIATE_ALL_METHODS(double)
INSTANTIATE_ALL_METHODS(std::int32_t)

#undef INSTANTIATE_ALL_HOST_METHODS
#undef INSTANTIATE_ALL_METHODS

} // namespace oneapi::dal::backend::interop
