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

#include "oneapi/dal/table/backend/interop/common.hpp"
#include "oneapi/dal/table/backend/interop/table_conversion.hpp"
#include "oneapi/dal/table/backend/interop/table_conversion_common.hpp"

namespace oneapi::dal::backend::interop {

template <typename Data>
homogen_table_ptr<Data> empty_daal_homogen_table(std::int64_t column_count) {
    return daal::data_management::HomogenNumericTable<Data>::create(
        dal::detail::integral_cast<std::size_t>(column_count),
        dal::detail::integral_cast<std::size_t>(0),
        daal::data_management::NumericTable::notAllocate);
}

template <typename Data>
homogen_table_ptr<Data> allocate_daal_homogen_table(std::int64_t row_count,
                                                    std::int64_t column_count) {
    return daal::data_management::HomogenNumericTable<Data>::create(
        dal::detail::integral_cast<std::size_t>(column_count),
        dal::detail::integral_cast<std::size_t>(row_count),
        daal::data_management::NumericTable::doAllocate);
}

template <typename Data>
homogen_table_ptr<Data> convert_to_daal_homogen_table(array<Data>& data,
                                                      std::int64_t row_count,
                                                      std::int64_t column_count,
                                                      bool allow_copy) {
    if (!data.get_count()) {
        return empty_daal_homogen_table<Data>(column_count);
    }

    if (allow_copy) {
        data.need_mutable_data();
    }

    ONEDAL_ASSERT(data.has_mutable_data());
    const auto daal_data = daal_shared_t<Data>( //
        data.get_mutable_data(),
        daal_object_owner{ data });

    return daal_homogen_t<Data>::create(daal_data,
                                        dal::detail::integral_cast<std::size_t>(column_count),
                                        dal::detail::integral_cast<std::size_t>(row_count));
}

template <typename Data>
homogen_table_ptr<Data> copy_to_daal_homogen_table(const table& table) {
    // TODO: Preserve information about features
    constexpr bool allow_copy = true;
    auto rows = row_accessor<const Data>{ table }.pull();
    return convert_to_daal_homogen_table(rows,
                                         table.get_row_count(),
                                         table.get_column_count(),
                                         allow_copy);
}

template <typename Data>
homogen_table convert_from_daal_homogen_table(const numeric_table_ptr& nt) {
    if (nt->getNumberOfRows() == 0) {
        return homogen_table{};
    }
    daal::data_management::BlockDescriptor<Data> block;
    const std::int64_t row_count = dal::detail::integral_cast<std::int64_t>(nt->getNumberOfRows());
    const std::int64_t column_count =
        dal::detail::integral_cast<std::int64_t>(nt->getNumberOfColumns());

    nt->getBlockOfRows(0, row_count, daal::data_management::readOnly, block);
    Data* data = block.getBlockPtr();
    array<Data> arr(data, row_count * column_count, [nt, block](Data* p) mutable {
        nt->releaseBlockOfRows(block);
    });

    return detail::homogen_table_builder{}.reset(arr, row_count, column_count).build();
}

numeric_table_ptr wrap_by_host_homogen_adapter(const homogen_table& table) {
    const auto dtype = table.get_metadata().get_data_type(0);

    const auto to_homogen = [&table](auto type_tag) -> numeric_table_ptr {
        using type_t = std::decay_t<decltype(type_tag)>;
        return host_homogen_table_adapter<type_t>::create(table);
    };

    const auto to_empty = [](auto dummy) -> numeric_table_ptr {
        return numeric_table_ptr{};
    };

    return dispatch_by_table_type(to_homogen, to_empty, dtype);
}

template <typename Data>
numeric_table_ptr convert_to_daal_table(const homogen_table& table) {
    if (table.get_data_layout() == data_layout::column_major) {
        if (auto wrapper = wrap_by_host_soa_adapter(table)) {
            return wrapper;
        }
    }
    else if (table.get_data_layout() == data_layout::row_major) {
        if (auto wrapper = wrap_by_host_homogen_adapter(table)) {
            return wrapper;
        }
    }
    return copy_to_daal_homogen_table<Data>(table);
}

template <typename Data>
csr_table_ptr convert_to_daal_csr_table(array<Data>& data,
                                        array<std::int64_t>& column_indices,
                                        array<std::int64_t>& row_indices,
                                        std::int64_t row_count,
                                        std::int64_t column_count,
                                        bool allow_copy) {
    ONEDAL_ASSERT(data.get_count() == column_indices.get_count());
    ONEDAL_ASSERT(row_indices.get_count() == row_count + 1);

    if (!data.get_count() || !column_indices.get_count() || !row_indices.get_count()) {
        return csr_table_ptr();
    }

    if (allow_copy) {
        data.need_mutable_data();
        column_indices.need_mutable_data();
        row_indices.need_mutable_data();
    }

    ONEDAL_ASSERT(data.has_mutable_data());
    ONEDAL_ASSERT(column_indices.has_mutable_data());
    ONEDAL_ASSERT(row_indices.has_mutable_data());

    const auto daal_data =
        daal::services::SharedPtr<Data>(data.get_mutable_data(), daal_object_owner{ data });
    ONEDAL_ASSERT(sizeof(std::size_t) == sizeof(std::int64_t));
    const auto daal_column_indices = daal::services::SharedPtr<std::size_t>(
        reinterpret_cast<std::size_t*>(column_indices.get_mutable_data()),
        daal_object_owner{ column_indices });
    const auto daal_row_indices = daal::services::SharedPtr<std::size_t>(
        reinterpret_cast<std::size_t*>(row_indices.get_mutable_data()),
        daal_object_owner{ row_indices });

    return daal_csr_t::create(daal_data,
                              daal_column_indices,
                              daal_row_indices,
                              dal::detail::integral_cast<std::size_t>(column_count),
                              dal::detail::integral_cast<std::size_t>(row_count));
}

template <typename Data>
csr_table_ptr copy_to_daal_csr_table(const csr_table& table) {
    constexpr bool allow_copy = true;
    auto [data, column_indices, row_offsets] = csr_accessor<const Data>{ table }.pull();
    return convert_to_daal_csr_table(data,
                                     column_indices,
                                     row_offsets,
                                     table.get_row_count(),
                                     table.get_column_count(),
                                     allow_copy);
}

csr_table_ptr wrap_by_host_csr_adapter(const csr_table& table) {
    const auto dtype = table.get_metadata().get_data_type(0);

    const auto to_csr = [&table](auto type_tag) -> csr_table_ptr {
        using type_t = std::decay_t<decltype(type_tag)>;
        return host_csr_table_adapter<type_t>::create(table);
    };

    const auto to_empty = [](auto dummy) -> csr_table_ptr {
        return csr_table_ptr{};
    };

    return dispatch_by_table_type(to_csr, to_empty, dtype);
}

template <typename Data>
csr_table_ptr convert_to_daal_table(const csr_table& table) {
    if (auto wrapper = wrap_by_host_csr_adapter(table)) {
        return wrapper;
    }
    else {
        return copy_to_daal_csr_table<Data>(table);
    }
}

template <typename Data>
csr_table convert_from_daal_csr_table(const numeric_table_ptr& nt) {
    auto block_owner = std::make_shared<csr_block_owner<Data>>(csr_block_owner<Data>{ nt });

    ONEDAL_ASSERT(sizeof(std::size_t) == sizeof(std::int64_t));

    return csr_table{
        array<Data>{ block_owner->get_data(),
                     block_owner->get_element_count(),
                     [block_owner](const Data* p) {} },
        array<std::int64_t>{ reinterpret_cast<std::int64_t*>(block_owner->get_column_indices()),
                             block_owner->get_element_count(),
                             [block_owner](const std::int64_t* p) {} },
        array<std::int64_t>{ reinterpret_cast<std::int64_t*>(block_owner->get_row_indices()),
                             block_owner->get_row_count() + 1,
                             [block_owner](const std::int64_t* p) {} },
        block_owner->get_column_count()
    };
}

soa_table_ptr wrap_by_host_soa_adapter(const homogen_table& table) {
    const auto dtype = table.get_metadata().get_data_type(0);

    const auto to_soa = [&table](auto type_tag) -> soa_table_ptr {
        using type_t = std::decay_t<decltype(type_tag)>;
        return host_soa_table_adapter::create<type_t>(table);
    };

    const auto to_empty = [](auto dummy) -> soa_table_ptr {
        return soa_table_ptr{};
    };

    return dispatch_by_table_type(to_soa, to_empty, dtype);
}

soa_table_ptr wrap_by_host_soa_adapter(const heterogen_table& table) {
    if (table.has_data()) {
        return host_heterogen_table_adapter::create(table);
    }
    else {
        return soa_table_ptr{};
    }
}

template <typename Data>
numeric_table_ptr convert_to_daal_table(const heterogen_table& table) {
    if (auto wrapper = wrap_by_host_soa_adapter(table)) {
        return wrapper;
    }
    return copy_to_daal_homogen_table<Data>(table);
}

heterogen_table convert_from_daal_table(const numeric_table_ptr& table) {
    auto& raw = static_cast<soa_table_t&>(*table);
    return convert_to_heterogen(raw);
}

template <typename Data>
table convert_from_daal_table(const numeric_table_ptr& nt) {
    using StorageLayout = dm::NumericTableIface::StorageLayout;
    if (nt->getDataLayout() == StorageLayout::csrArray) {
        return convert_from_daal_csr_table<Data>(nt);
    }
    else if (nt->getDataLayout() == StorageLayout::aos) {
        return convert_from_daal_homogen_table<Data>(nt);
    }
    else if (nt->getDataLayout() == StorageLayout::soa) {
        return convert_from_daal_heterogen_table(nt);
    }
    else {
        return table{};
    }
}

template <typename Data>
numeric_table_ptr convert_to_daal_table(const table& table) {
    if (table.get_kind() == homogen_table::kind()) {
        const auto& homogen = static_cast<const homogen_table&>(table);
        return convert_to_daal_table<Data>(homogen);
    }
    else if (table.get_kind() == heterogen_table::kind()) {
        const auto& heterogen = static_cast<const heterogen_table&>(table);
        return convert_to_daal_table<Data>(heterogen);
    }
    else if (table.get_kind() == csr_table::kind()) {
        const auto& csr = static_cast<const csr_table&>(table);
        return convert_to_daal_table<Data>(csr);
    }
    else {
        return copy_to_daal_homogen_table<Data>(table);
    }
}

#define INSTANTIATE(TYPE)                                                                          \
    template ONEDAL_EXPORT numeric_table_ptr convert_to_daal_table<TYPE>(const homogen_table&);    \
    template ONEDAL_EXPORT homogen_table_ptr<TYPE> copy_to_daal_homogen_table<TYPE>(const table&); \
    template ONEDAL_EXPORT homogen_table_ptr<TYPE> convert_to_daal_homogen_table(array<TYPE>&,     \
                                                                                 std::int64_t,     \
                                                                                 std::int64_t,     \
                                                                                 bool);            \
    template ONEDAL_EXPORT csr_table_ptr convert_to_daal_csr_table(array<TYPE>&,                   \
                                                                   array<std::int64_t>&,           \
                                                                   array<std::int64_t>&,           \
                                                                   std::int64_t,                   \
                                                                   std::int64_t,                   \
                                                                   bool);                          \
    template ONEDAL_EXPORT csr_table_ptr copy_to_daal_csr_table<TYPE>(const csr_table&);           \
    template ONEDAL_EXPORT csr_table_ptr convert_to_daal_table<TYPE>(const csr_table&);            \
    template ONEDAL_EXPORT csr_table convert_from_daal_csr_table<TYPE>(const numeric_table_ptr&);  \
    template ONEDAL_EXPORT numeric_table_ptr convert_to_daal_table<TYPE>(const heterogen_table&);  \
    template ONEDAL_EXPORT table convert_from_daal_table<TYPE>(const numeric_table_ptr&);          \
    template ONEDAL_EXPORT numeric_table_ptr convert_to_daal_table<TYPE>(const table& table);

INSTANTIATE(std::int32_t)
INSTANTIATE(double)
INSTANTIATE(float)

} // namespace oneapi::dal::backend::interop
