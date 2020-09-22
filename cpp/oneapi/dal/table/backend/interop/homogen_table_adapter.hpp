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

#include <limits>

#include <daal/include/data_management/data/homogen_numeric_table.h>

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/column_accessor.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/daal_array_owner.hpp"

namespace oneapi::dal::backend::interop {

// This class shall be used only to represent immutable data on DAAL side.
// Any attempts to change the data inside objects of that class lead to undefined behavior.
template <typename Policy, typename Data>
class homogen_table_adapter : public daal::data_management::HomogenNumericTable<Data> {
    static constexpr bool is_host_only = std::is_same_v<Policy, detail::default_host_policy>;

#ifdef ONEAPI_DAL_DATA_PARALLEL
    static_assert(
        detail::is_one_of<Policy, detail::default_host_policy, detail::data_parallel_policy>::value,
        "Adapter supports only default_host_policy and data_parallel_policy");
#else
    static_assert(is_host_only, "Adapter supports only default_host_policy");
#endif

    using base = daal::data_management::HomogenNumericTable<Data>;
    using ptr_t = daal::services::SharedPtr<homogen_table_adapter>;
    using ptr_data_t = daal::services::SharedPtr<Data>;
    using status_t = daal::services::Status;
    using rw_mode_t = daal::data_management::ReadWriteMode;

    template <typename T>
    using block_desc_t = daal::data_management::BlockDescriptor<T>;

public:
    static ptr_t create(const Policy& policy, const homogen_table& table) {
        status_t internal_stat;
        auto result = ptr_t{ new homogen_table_adapter(policy, table, internal_stat) };
        status_to_exception(internal_stat);
        return result;
    }

private:
    class block_info {
    public:
        template <typename BlockData>
        block_info(const block_desc_t<BlockData>& block,
                   std::size_t row_begin_index,
                   std::size_t value_count,
                   std::size_t column_index)
                : block_info(block, row_begin_index, value_count) {
            DAAL_ASSERT(column_index < std::numeric_limits<std::int64_t>::max());
            this->column_index = static_cast<std::int64_t>(column_index);
        }

        template <typename BlockData>
        block_info(const block_desc_t<BlockData>& block,
                   std::size_t row_begin_index,
                   std::size_t row_count) {
            DAAL_ASSERT(block.getNumberOfRows() < std::numeric_limits<std::int64_t>::max());
            DAAL_ASSERT(block.getNumberOfColumns() < std::numeric_limits<std::int64_t>::max());
            DAAL_ASSERT(row_begin_index < std::numeric_limits<std::int64_t>::max());
            DAAL_ASSERT(row_count < std::numeric_limits<std::int64_t>::max());

            const auto block_desc_row_count = static_cast<std::int64_t>(block.getNumberOfRows());
            const auto block_desc_column_count =
                static_cast<std::int64_t>(block.getNumberOfColumns());

            this->size = block_desc_row_count * block_desc_column_count; // TODO: check overflow
            this->row_begin_index = static_cast<std::int64_t>(row_begin_index);
            this->row_count = static_cast<std::int64_t>(row_count);
            this->row_end_index = this->row_begin_index + this->row_count; // TODO: check overflow
            this->column_index = -1;
        }

        std::int64_t size;
        std::int64_t column_index;
        std::int64_t row_begin_index;
        std::int64_t row_end_index;
        std::int64_t row_count;
        std::int64_t range_size;
    };

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
        DAAL_ASSERT(!"homogen_table_adapter: getSerializationTag() is not implemented");
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

        const std::int64_t column_count = original_table_.get_column_count();
        const block_info info{ block, vector_idx, vector_num };

        if (check_row_indexes_in_range(info) == false) {
            return status_t(daal::services::ErrorIncorrectIndex);
        }

        if constexpr (std::is_same_v<BlockData, Data>) {
            auto raw_data_ptr = this->_ptr.get() + column_count * vector_idx * sizeof(Data);

            if constexpr (is_host_only) {
                block.setPtr(&this->_ptr, raw_data_ptr, column_count, info.row_count);
            }
#ifdef ONEAPI_DAL_DATA_PARALLEL
            else {
                daal::services::Buffer<BlockData> buffer(reinterpret_cast<BlockData*>(raw_data_ptr),
                                                         info.row_count * column_count,
                                                         data_kind_);
                // this operation is safe only when the table does not leave the scope
                // before the block. Otherwise block contains dangling pointer.
                block.setBuffer(buffer, column_count, info.row_count);
            }
#endif
        }
        else {
            try {
                array<BlockData> values;
                auto block_ptr = block.getBlockPtr();
                // TODO: check overflow below
                if (block_ptr != nullptr && info.size >= info.row_count * column_count) {
                    values.reset(block_ptr, info.size, dal::empty_delete<BlockData>());
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

        const block_info info{ block, vector_idx, value_num, feature_idx };

        if (check_row_indexes_in_range(info) == false ||
            check_column_index_in_range(info) == false) {
            return status_t(daal::services::ErrorIncorrectIndex);
        }

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

        return status_t();
    }

    bool check_row_indexes_in_range(const block_info& info) const {
        const std::int64_t row_count = original_table_.get_row_count();
        return info.row_begin_index >= 0 && info.row_begin_index < row_count &&
               info.row_end_index > info.row_begin_index && info.row_end_index <= row_count;
    }

    bool check_column_index_in_range(const block_info& info) const {
        const std::int64_t column_count = original_table_.get_column_count();
        return info.column_index >= 0 && info.column_index < column_count;
    }

    template <typename Accessor, typename BlockData, typename... Args>
    void pull_values(block_desc_t<BlockData>& block,
                     std::int64_t row_count,
                     std::int64_t column_count,
                     const Accessor& acc,
                     array<BlockData>& values,
                     Args&&... args) const {
        // The following const_cast operations are safe only when this class is used for read-only
        // operations. Use on write leads to undefined behaviour.

        if constexpr (is_host_only) {
            if (block.getBlockPtr() != acc.pull(values, std::forward<Args>(args)...)) {
                auto raw_ptr = const_cast<BlockData*>(values.get_data());
                auto data_shared =
                    daal::services::SharedPtr<BlockData>(raw_ptr, daal_array_owner(values));
                block.setSharedPtr(data_shared, column_count, row_count);
            }
        }
#ifdef ONEAPI_DAL_DATA_PARALLEL
        else {
            auto values_data_kind =
                values.get_count() > 0
                    ? sycl::get_pointer_type(values.get_data(), policy_.get_queue().get_context())
                    : data_kind_;
            if (block.getBlockPtr() != acc.pull(policy_.get_queue(),
                                                values,
                                                std::forward<Args>(args)...,
                                                values_data_kind)) {
                auto raw_ptr = const_cast<BlockData*>(values.get_data());
                daal::services::Buffer<BlockData> buffer(raw_ptr,
                                                         row_count * column_count,
                                                         values_data_kind);
                block.setBuffer(buffer, column_count, row_count);
            }
        }
#endif
    }

    void setup_data_kind(const homogen_table& table) {
        if constexpr (is_host_only) {
            return;
        }
#ifdef ONEAPI_DAL_DATA_PARALLEL
        else {
            auto data = table.get_data();
            data_kind_ = sycl::get_pointer_type(data, policy_.get_queue().get_context());
        }
#endif
    }

    homogen_table_adapter(const Policy& policy, const homogen_table& table, status_t& stat)
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
              policy_(policy) {
        if (stat.ok() == false) {
            return;
        }
        else if (table.has_data() == false || table.get_data_layout() != data_layout::row_major) {
            stat.add(daal::services::ErrorIncorrectParameter);
            return;
        }

        original_table_ = table;
        setup_data_kind(original_table_);

        this->_memStatus = daal::data_management::NumericTableIface::userAllocated;
        this->_layout = daal::data_management::NumericTableIface::aos;
    }

private:
    Policy policy_;
    homogen_table original_table_;

#ifdef ONEAPI_DAL_DATA_PARALLEL
    sycl::usm::alloc data_kind_;
#endif
};

} // namespace oneapi::dal::backend::interop
