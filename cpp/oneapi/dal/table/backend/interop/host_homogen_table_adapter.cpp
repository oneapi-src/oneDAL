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

#include "oneapi/dal/table/backend/interop/host_homogen_table_adapter.hpp"

namespace oneapi::dal::backend::interop {

namespace daal_dm = daal::data_management;

static daal_dm::features::FeatureType get_daal_feature_type(feature_type t) {
    switch (t) {
        case feature_type::nominal: return daal_dm::features::DAAL_CATEGORICAL;
        case feature_type::ordinal: return daal_dm::features::DAAL_ORDINAL;
        case feature_type::interval: return daal_dm::features::DAAL_CONTINUOUS;
        case feature_type::ratio: return daal_dm::features::DAAL_CONTINUOUS;
        default: throw dal::internal_error(detail::error_messages::unsupported_feature_type());
    }
}

static void convert_feature_information_to_daal(const table_metadata& src,
                                                daal_dm::NumericTableDictionary& dst) {
    ONEDAL_ASSERT(std::size_t(src.get_feature_count()) == dst.getNumberOfFeatures());
    for (std::int64_t i = 0; i < src.get_feature_count(); i++) {
        auto& daal_feature = dst[i];
        daal_feature.featureType = get_daal_feature_type(src.get_feature_type(i));
    }
}

template <typename Accessor, typename BlockData, typename... Args>
static void pull_values(daal_dm::BlockDescriptor<BlockData>& block,
                        std::int64_t row_count,
                        std::int64_t column_count,
                        const Accessor& acc,
                        array<BlockData>& values,
                        Args&&... args) {
    // The following const_cast operation is safe only when this class is used
    // for read-only operations. Use on write leads to undefined behaviour.
    if (block.getBlockPtr() != acc.pull(values, std::forward<Args>(args)...)) {
        auto raw_ptr = const_cast<BlockData*>(values.get_data());
        auto data_shared = daal::services::SharedPtr<BlockData>(raw_ptr, daal_object_owner(values));
        block.setSharedPtr(data_shared, column_count, row_count);
    }
}

template <typename Data>
auto host_homogen_table_adapter<Data>::create(const homogen_table& table) -> ptr_t {
    status_t internal_stat;
    auto result = ptr_t{ new host_homogen_table_adapter(table, internal_stat) };
    status_to_exception(internal_stat);
    return result;
}

template <typename Data>
host_homogen_table_adapter<Data>::host_homogen_table_adapter(const homogen_table& table,
                                                             status_t& status)
        // The following const_cast is safe only when this class is used for read-only
        // operations. Use on write leads to undefined behaviour.
        : base(daal_dm::DictionaryIface::equal,
               ptr_data_t{ const_cast<Data*>(table.get_data<Data>()), daal_object_owner(table) },
               table.get_column_count(),
               table.get_row_count(),
               status),
          is_rowmajor_(table.get_data_layout() == data_layout::row_major),
          original_table_(table) {
    if (!status.ok()) {
        return;
    }
    else if (!table.has_data()) {
        status.add(daal::services::ErrorIncorrectParameter);
        return;
    }

    this->_memStatus = daal_dm::NumericTableIface::userAllocated;
    this->_layout = daal_dm::NumericTableIface::aos;

    convert_feature_information_to_daal(original_table_.get_metadata(),
                                        *this->getDictionarySharedPtr());
}

template <typename Data>
auto host_homogen_table_adapter<Data>::getBlockOfRows(std::size_t vector_idx,
                                                      std::size_t vector_num,
                                                      rw_mode_t rwflag,
                                                      block_desc_t<double>& block) -> status_t {
    return convert_exception_to_status([&]() {
        return read_rows_impl(vector_idx, vector_num, rwflag, block);
    });
}

template <typename Data>
auto host_homogen_table_adapter<Data>::getBlockOfRows(std::size_t vector_idx,
                                                      std::size_t vector_num,
                                                      rw_mode_t rwflag,
                                                      block_desc_t<float>& block) -> status_t {
    return convert_exception_to_status([&]() {
        return read_rows_impl(vector_idx, vector_num, rwflag, block);
    });
}

template <typename Data>
auto host_homogen_table_adapter<Data>::getBlockOfRows(std::size_t vector_idx,
                                                      std::size_t vector_num,
                                                      rw_mode_t rwflag,
                                                      block_desc_t<int>& block) -> status_t {
    return convert_exception_to_status([&]() {
        return read_rows_impl(vector_idx, vector_num, rwflag, block);
    });
}

template <typename Data>
auto host_homogen_table_adapter<Data>::getBlockOfColumnValues(std::size_t feature_idx,
                                                              std::size_t vector_idx,
                                                              std::size_t value_num,
                                                              rw_mode_t rwflag,
                                                              block_desc_t<double>& block)
    -> status_t {
    return convert_exception_to_status([&]() {
        return read_column_values_impl(feature_idx, vector_idx, value_num, rwflag, block);
    });
}

template <typename Data>
auto host_homogen_table_adapter<Data>::getBlockOfColumnValues(std::size_t feature_idx,
                                                              std::size_t vector_idx,
                                                              std::size_t value_num,
                                                              rw_mode_t rwflag,
                                                              block_desc_t<float>& block)
    -> status_t {
    return convert_exception_to_status([&]() {
        return read_column_values_impl(feature_idx, vector_idx, value_num, rwflag, block);
    });
}

template <typename Data>
auto host_homogen_table_adapter<Data>::getBlockOfColumnValues(std::size_t feature_idx,
                                                              std::size_t vector_idx,
                                                              std::size_t value_num,
                                                              rw_mode_t rwflag,
                                                              block_desc_t<int>& block)
    -> status_t {
    return convert_exception_to_status([&]() {
        return read_column_values_impl(feature_idx, vector_idx, value_num, rwflag, block);
    });
}

template <typename Data>
auto host_homogen_table_adapter<Data>::releaseBlockOfRows(block_desc_t<double>& block) -> status_t {
    block.reset();
    return status_t();
}

template <typename Data>
auto host_homogen_table_adapter<Data>::releaseBlockOfRows(block_desc_t<float>& block) -> status_t {
    block.reset();
    return status_t();
}

template <typename Data>
auto host_homogen_table_adapter<Data>::releaseBlockOfRows(block_desc_t<int>& block) -> status_t {
    block.reset();
    return status_t();
}

template <typename Data>
auto host_homogen_table_adapter<Data>::releaseBlockOfColumnValues(block_desc_t<double>& block)
    -> status_t {
    block.reset();
    return status_t();
}

template <typename Data>
auto host_homogen_table_adapter<Data>::releaseBlockOfColumnValues(block_desc_t<float>& block)
    -> status_t {
    block.reset();
    return status_t();
}

template <typename Data>
auto host_homogen_table_adapter<Data>::releaseBlockOfColumnValues(block_desc_t<int>& block)
    -> status_t {
    block.reset();
    return status_t();
}

template <typename Data>
auto host_homogen_table_adapter<Data>::assign(float) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

template <typename Data>
auto host_homogen_table_adapter<Data>::assign(double) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

template <typename Data>
auto host_homogen_table_adapter<Data>::assign(int) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

template <typename Data>
auto host_homogen_table_adapter<Data>::allocateDataMemoryImpl(daal::MemType) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

template <typename Data>
auto host_homogen_table_adapter<Data>::setNumberOfColumnsImpl(std::size_t) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

template <typename Data>
int host_homogen_table_adapter<Data>::getSerializationTag() const {
    ONEDAL_ASSERT(!"host_homogen_table_adapter: getSerializationTag() is not implemented");
    return -1;
}

template <typename Data>
auto host_homogen_table_adapter<Data>::serializeImpl(daal_dm::InputDataArchive* arch) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

template <typename Data>
auto host_homogen_table_adapter<Data>::deserializeImpl(const daal_dm::OutputDataArchive* arch)
    -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

template <typename Data>
void host_homogen_table_adapter<Data>::freeDataMemoryImpl() {
    base::freeDataMemoryImpl();
    original_table_ = homogen_table{};
}

template <typename Data>
template <typename BlockData>
auto host_homogen_table_adapter<Data>::read_rows_impl(std::size_t vector_idx,
                                                      std::size_t vector_num,
                                                      rw_mode_t rwflag,
                                                      block_desc_t<BlockData>& block) -> status_t {
    if (rwflag != daal_dm::readOnly) {
        ONEDAL_ASSERT(!"Data is accessible in read-only mode by design");
        return daal::services::ErrorMethodNotImplemented;
    }

    if (is_rowmajor_) {
        return base::getBlockOfRows(vector_idx, vector_num, rwflag, block);
    }
    else {
        const std::int64_t column_count = original_table_.get_column_count();
        const block_info info{ block, vector_idx, vector_num };

        if (!check_row_indexes_in_range(info)) {
            return daal::services::ErrorIncorrectIndex;
        }

        block.setDetails(0, vector_idx, rwflag);

        array<BlockData> values;
        auto block_ptr = block.getBlockPtr();

        // multiplication is safe due to checks with 'info' variable
        const std::int64_t requested_element_count = info.row_count * column_count;

        if (block_ptr != nullptr && info.allocated_element_count >= requested_element_count) {
            values.reset(block_ptr,
                         info.allocated_element_count,
                         detail::empty_delete<BlockData>());
        }

        const row_accessor<const BlockData> acc{ original_table_ };
        pull_values(block,
                    info.row_count,
                    column_count,
                    acc,
                    values,
                    range{ info.row_begin_index, info.row_end_index });
    }

    return status_t();
}

template <typename Data>
template <typename BlockData>
auto host_homogen_table_adapter<Data>::read_column_values_impl(std::size_t feature_idx,
                                                               std::size_t vector_idx,
                                                               std::size_t value_num,
                                                               rw_mode_t rwflag,
                                                               block_desc_t<BlockData>& block)
    -> status_t {
    if (rwflag != daal_dm::readOnly) {
        ONEDAL_ASSERT(!"Data is accessible in read-only mode by design");
        return daal::services::ErrorMethodNotImplemented;
    }

    if (is_rowmajor_) {
        return base::getBlockOfColumnValues(feature_idx, vector_idx, value_num, rwflag, block);
    }
    else {
        const block_info info{ block, vector_idx, value_num, feature_idx };

        if (!check_row_indexes_in_range(info) || !check_column_index_in_range(info)) {
            return daal::services::ErrorIncorrectIndex;
        }

        block.setDetails(feature_idx, vector_idx, rwflag);

        array<BlockData> values;
        auto block_ptr = block.getBlockPtr();
        if (block_ptr != nullptr && info.allocated_element_count >= info.row_count) {
            values.reset(block_ptr,
                         info.allocated_element_count,
                         detail::empty_delete<BlockData>());
        }

        const column_accessor<const BlockData> acc{ original_table_ };
        pull_values(block,
                    info.row_count,
                    1,
                    acc,
                    values,
                    info.column_index,
                    range{ info.row_begin_index, info.row_end_index });
    }
    return status_t();
}

template class host_homogen_table_adapter<std::int32_t>;
template class host_homogen_table_adapter<float>;
template class host_homogen_table_adapter<double>;

} // namespace oneapi::dal::backend::interop
