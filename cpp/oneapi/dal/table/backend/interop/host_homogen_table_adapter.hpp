/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <daal/include/data_management/data/homogen_numeric_table.h>

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/column_accessor.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/daal_array_owner.hpp" //create
#include "oneapi/dal/table/backend/interop/block_info.hpp"

namespace oneapi::dal::backend::interop {

// This class shall be used only to represent immutable data on DAAL side.
// Any attempts to change the data inside objects of that class lead to undefined behavior.
template <typename Data>
class host_homogen_table_adapter : public daal::data_management::HomogenNumericTable<Data> {
    using base = daal::data_management::HomogenNumericTable<Data>;
    using ptr_t = daal::services::SharedPtr<host_homogen_table_adapter>;
    using ptr_data_t = daal::services::SharedPtr<Data>;
    using status_t = daal::services::Status;
    using rw_mode_t = daal::data_management::ReadWriteMode;

    template <typename T>
    using block_desc_t = daal::data_management::BlockDescriptor<T>;

public:
    static ptr_t create(const homogen_table& table) {
        status_t internal_stat;
        auto result = ptr_t{ new host_homogen_table_adapter(table, internal_stat) };
        status_to_exception(internal_stat);
        return result;
    }

private:
    status_t getBlockOfRows(std::size_t vector_idx,
                            std::size_t vector_num,
                            rw_mode_t rwflag,
                            block_desc_t<double>& block) override {
        return read_rows_impl(vector_idx, vector_num, rwflag, block);
    }
    status_t getBlockOfRows(std::size_t vector_idx,
                            std::size_t vector_num,
                            rw_mode_t rwflag,
                            block_desc_t<float>& block) override {
        return read_rows_impl(vector_idx, vector_num, rwflag, block);
    }
    status_t getBlockOfRows(std::size_t vector_idx,
                            std::size_t vector_num,
                            rw_mode_t rwflag,
                            block_desc_t<int>& block) override {
        return read_rows_impl(vector_idx, vector_num, rwflag, block);
    }

    status_t getBlockOfColumnValues(std::size_t feature_idx,
                                    std::size_t vector_idx,
                                    std::size_t value_num,
                                    rw_mode_t rwflag,
                                    block_desc_t<double>& block) override {
        return read_column_values_impl(feature_idx, vector_idx, value_num, rwflag, block);
    }
    status_t getBlockOfColumnValues(std::size_t feature_idx,
                                    std::size_t vector_idx,
                                    std::size_t value_num,
                                    rw_mode_t rwflag,
                                    block_desc_t<float>& block) override {
        return read_column_values_impl(feature_idx, vector_idx, value_num, rwflag, block);
    }
    status_t getBlockOfColumnValues(std::size_t feature_idx,
                                    std::size_t vector_idx,
                                    std::size_t value_num,
                                    rw_mode_t rwflag,
                                    block_desc_t<int>& block) override {
        return read_column_values_impl(feature_idx, vector_idx, value_num, rwflag, block);
    }

    status_t releaseBlockOfRows(block_desc_t<double>& block) override {
        block.reset();
        return status_t();
    }
    status_t releaseBlockOfRows(block_desc_t<float>& block) override {
        block.reset();
        return status_t();
    }
    status_t releaseBlockOfRows(block_desc_t<int>& block) override {
        block.reset();
        return status_t();
    }

    status_t releaseBlockOfColumnValues(block_desc_t<double>& block) override {
        block.reset();
        return status_t();
    }
    status_t releaseBlockOfColumnValues(block_desc_t<float>& block) override {
        block.reset();
        return status_t();
    }
    status_t releaseBlockOfColumnValues(block_desc_t<int>& block) override {
        block.reset();
        return status_t();
    }

    status_t assign(float value) override {
        return status_t(daal::services::ErrorMethodNotImplemented);
    }

    status_t assign(double value) override {
        return status_t(daal::services::ErrorMethodNotImplemented);
    }

    status_t assign(int value) override {
        return status_t(daal::services::ErrorMethodNotImplemented);
    }

    status_t allocateDataMemoryImpl(daal::MemType /*type*/ = daal::dram) override {
        return status_t(daal::services::ErrorMethodNotImplemented);
    }

    status_t setNumberOfColumnsImpl(std::size_t ncol) override {
        return status_t(daal::services::ErrorMethodNotImplemented);
    }

    int getSerializationTag() const override {
        DAAL_ASSERT(!"host_homogen_table_adapter: getSerializationTag() is not implemented");
        return -1;
    }

    status_t serializeImpl(daal::data_management::InputDataArchive* arch) override {
        return status_t(daal::services::ErrorMethodNotImplemented);
    }

    status_t deserializeImpl(const daal::data_management::OutputDataArchive* arch) override {
        return status_t(daal::services::ErrorMethodNotImplemented);
    }

    void freeDataMemoryImpl() override {
        base::freeDataMemoryImpl();
        original_table_ = homogen_table{};
    }

    template <typename BlockData>
    status_t read_rows_impl(std::size_t vector_idx,
                            std::size_t vector_num,
                            rw_mode_t rwflag,
                            block_desc_t<BlockData>& block) {
        if (rwflag != daal::data_management::readOnly) {
            return status_t(daal::services::ErrorMethodNotImplemented);
        }

        if (is_rowmajor_) {
            return base::getBlockOfRows(vector_idx, vector_num, rwflag, block);
        }
        else {
            const std::int64_t column_count = original_table_.get_column_count();
            const block_info info{ block, vector_idx, vector_num };

            if (check_row_indexes_in_range(info) == false) {
                return status_t(daal::services::ErrorIncorrectIndex);
            }

            block.setDetails(0, vector_idx, rwflag);

            try {
                array<BlockData> values;
                auto block_ptr = block.getBlockPtr();

                // multiplication is safe due to checks with 'info' variable
                const std::int64_t requested_element_count = info.row_count * column_count;

                if (block_ptr != nullptr && info.bd_element_count >= requested_element_count) {
                    values.reset(block_ptr, info.bd_element_count, dal::empty_delete<BlockData>());
                }

                const row_accessor<const BlockData> acc{ original_table_ };
                pull_values(block,
                            info.row_count,
                            column_count,
                            acc,
                            values,
                            range{ info.row_begin_index, info.row_end_index });
            }
            catch (const bad_alloc&) {
                return status_t(daal::services::ErrorMemoryAllocationFailed);
            }
            catch (const out_of_range&) {
                return status_t(daal::services::ErrorIncorrectDataRange);
            }
            catch (const std::exception&) {
                return status_t(daal::services::UnknownError);
            }
        }

        return status_t();
    }

    template <typename BlockData>
    status_t read_column_values_impl(std::size_t feature_idx,
                                     std::size_t vector_idx,
                                     std::size_t value_num,
                                     rw_mode_t rwflag,
                                     block_desc_t<BlockData>& block) {
        if (rwflag != daal::data_management::readOnly) {
            return status_t(daal::services::ErrorMethodNotImplemented);
        }

        if (is_rowmajor_) {
            return base::getBlockOfColumnValues(feature_idx, vector_idx, value_num, rwflag, block);
        }
        else {
            const block_info info{ block, vector_idx, value_num, feature_idx };

            if (check_row_indexes_in_range(info) == false ||
                check_column_index_in_range(info) == false) {
                return status_t(daal::services::ErrorIncorrectIndex);
            }

            block.setDetails(feature_idx, vector_idx, rwflag);

            try {
                array<BlockData> values;
                auto block_ptr = block.getBlockPtr();
                if (block_ptr != nullptr && info.size >= info.row_count) {
                    values.reset(block_ptr, info.size, dal::empty_delete<BlockData>());
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
            catch (const bad_alloc&) {
                return status_t(daal::services::ErrorMemoryAllocationFailed);
            }
            catch (const out_of_range&) {
                return status_t(daal::services::ErrorIncorrectDataRange);
            }
            catch (const std::exception&) {
                return status_t(daal::services::UnknownError);
            }
        }
        return status_t();
    }

    bool check_row_indexes_in_range(const block_info& info) const {
        const std::int64_t row_count = original_table_.get_row_count();
        return info.row_begin_index < row_count && info.row_end_index <= row_count;
    }

    bool check_column_index_in_range(const block_info& info) const {
        const std::int64_t column_count = original_table_.get_column_count();
        return info.single_column_requested && info.column_index < column_count;
    }

    template <typename Accessor, typename BlockData, typename... Args>
    void pull_values(block_desc_t<BlockData>& block,
                     std::int64_t row_count,
                     std::int64_t column_count,
                     const Accessor& acc,
                     array<BlockData>& values,
                     Args&&... args) const {
        // The following const_cast operation is safe only when this class is used for read-only
        // operations. Use on write leads to undefined behaviour.

        if (block.getBlockPtr() != acc.pull(values, std::forward<Args>(args)...)) {
            auto raw_ptr = const_cast<BlockData*>(values.get_data());
            auto data_shared =
                daal::services::SharedPtr<BlockData>(raw_ptr, daal_array_owner(values));
            block.setSharedPtr(data_shared, column_count, row_count);
        }
    }

    host_homogen_table_adapter(const homogen_table& table, status_t& stat)
            // The following const_cast is safe only when this class is used for read-only
            // operations. Use on write leads to undefined behaviour.
            : base(daal::data_management::DictionaryIface::equal,
                   ptr_data_t{ const_cast<Data*>(table.get_data<Data>()),
                               [data_owner = homogen_table{ table }](auto ptr) mutable {
                                   data_owner = homogen_table{};
                               } },
                   table.get_column_count(),
                   table.get_row_count(),
                   stat),
             is_rowmajor_(table.get_data_layout() == data_layout::row_major) {
        if (stat.ok() == false) {
            return;
        }
        else if (table.has_data() == false) {
            stat.add(daal::services::ErrorIncorrectParameter);
            return;
        }

        original_table_ = table;

        this->_memStatus = daal::data_management::NumericTableIface::userAllocated;
        this->_layout = daal::data_management::NumericTableIface::aos;
    }

private:
    const bool is_rowmajor_;
    homogen_table original_table_;
};

} // namespace oneapi::dal::backend::interop
