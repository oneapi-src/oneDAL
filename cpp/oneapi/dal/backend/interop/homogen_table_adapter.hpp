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

#include <daal/include/data_management/data/homogen_numeric_table.h>

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/column_accessor.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"

namespace oneapi::dal::backend::interop {

template <typename Policy, typename Data>
class homogen_table_adapter : public daal::data_management::HomogenNumericTable<Data> {

    static constexpr bool is_host_only = std::is_same_v<Policy, detail::default_host_policy>;

#ifdef ONEAPI_DAL_DATA_PARALLEL
    static_assert(detail::is_one_of<Policy, detail::default_host_policy, detail::data_parallel_policy>::value,
                  "Adapter supports only host/dpc++ policies");
#else
    static_assert(is_host_only,
                  "Adapter supports only default host policy");
#endif

    using base = daal::data_management::HomogenNumericTable<Data>;
    using ptr_t = daal::services::SharedPtr<homogen_table_adapter>;
    using status_t = daal::services::Status;
    using rw_mode_t = daal::data_management::ReadWriteMode;

    template <typename T>
    using block_desc_t = daal::data_management::BlockDescriptor<T>;

public:
    static ptr_t create(const Policy& policy, const homogen_table& table) {
        status_t internal_stat;
        auto result = ptr_t { new homogen_table_adapter(policy, table, internal_stat) };
        status_to_exception(internal_stat);
        return result;
    }

    ~homogen_table_adapter() { freeDataMemoryImpl(); }

private:
    class block_info {
    public:
        template <typename Data>
        block_info(const block_desc_t<Data>& block,
                   size_t row_begin_index,
                   size_t value_count,
                   size_t column_index)
            : block_info(block, row_begin_index, value_count) {
            this->column_index = static_cast<std::int64_t>(column_index);   // TODO: check overflow
        }

        template <typename Data>
        block_info(const block_desc_t<Data>& block,
                   size_t row_begin_index,
                   size_t row_count) {
            const auto block_desc_rows = static_cast<std::int64_t>(block.getNumberOfRows());
            const auto block_desc_cols = static_cast<std::int64_t>(block.getNumberOfColumns());

            this->size = block_desc_rows*block_desc_cols;
            this->row_begin_index = static_cast<std::int64_t>(row_begin_index); // TODO: check overflow
            this->row_count = static_cast<std::int64_t>(row_count); // TODO: check overflow
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

    status_t getBlockOfRows(size_t vector_idx, size_t vector_num, rw_mode_t rwflag, block_desc_t<double> & block) DAAL_C11_OVERRIDE {
        return readRows(vector_idx, vector_num, rwflag, block);
    }
    status_t getBlockOfRows(size_t vector_idx, size_t vector_num, rw_mode_t rwflag, block_desc_t<float> & block) DAAL_C11_OVERRIDE {
        return readRows(vector_idx, vector_num, rwflag, block);
    }
    status_t getBlockOfRows(size_t vector_idx, size_t vector_num, rw_mode_t rwflag, block_desc_t<int> & block) DAAL_C11_OVERRIDE {
        return readRows(vector_idx, vector_num, rwflag, block);
    }

    status_t getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, rw_mode_t rwflag,
                                    block_desc_t<double> & block) DAAL_C11_OVERRIDE {
        return readColumnValues(feature_idx, vector_idx, value_num, rwflag, block);
    }
    status_t getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, rw_mode_t rwflag,
                                    block_desc_t<float> & block) DAAL_C11_OVERRIDE {
        return readColumnValues(feature_idx, vector_idx, value_num, rwflag, block);
    }
    status_t getBlockOfColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, rw_mode_t rwflag,
                                    block_desc_t<int> & block) DAAL_C11_OVERRIDE {
        return readColumnValues(feature_idx, vector_idx, value_num, rwflag, block);
    }

    status_t releaseBlockOfRows(block_desc_t<double> & block) DAAL_C11_OVERRIDE {
        block.reset();
        return status_t();
    }
    status_t releaseBlockOfRows(block_desc_t<float> & block) DAAL_C11_OVERRIDE {
        block.reset();
        return status_t();
    }
    status_t releaseBlockOfRows(block_desc_t<int> & block) DAAL_C11_OVERRIDE {
        block.reset();
        return status_t();
    }

    status_t releaseBlockOfColumnValues(block_desc_t<double> & block) DAAL_C11_OVERRIDE {
        block.reset();
        return status_t();
    }
    status_t releaseBlockOfColumnValues(block_desc_t<float> & block) DAAL_C11_OVERRIDE {
        block.reset();
        return status_t();
    }
    status_t releaseBlockOfColumnValues(block_desc_t<int> & block) DAAL_C11_OVERRIDE {
        block.reset();
        return status_t();
    }

    status_t assign(float value) DAAL_C11_OVERRIDE {
        return status_t(daal::services::ErrorMethodNotSupported);
    }

    status_t assign(double value) DAAL_C11_OVERRIDE {
        return status_t(daal::services::ErrorMethodNotSupported);
    }

    status_t assign(int value) DAAL_C11_OVERRIDE {
        return status_t(daal::services::ErrorMethodNotSupported);
    }

    status_t allocateDataMemoryImpl(daal::MemType /*type*/ = daal::dram) DAAL_C11_OVERRIDE {
        return status_t(daal::services::ErrorMethodNotSupported);
    }

    status_t setNumberOfColumnsImpl(size_t ncol) DAAL_C11_OVERRIDE {
        return status_t(daal::services::ErrorMethodNotSupported);
    }

    virtual int getSerializationTag() const DAAL_C11_OVERRIDE {
        DAAL_ASSERT(!"homogen_table_adapter: getSerializationTag() is not implemented");
        return -1;
    }

    status_t serializeImpl(daal::data_management::InputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        return status_t(daal::services::ErrorMethodNotSupported);
    }

    status_t deserializeImpl(const daal::data_management::OutputDataArchive * arch) DAAL_C11_OVERRIDE
    {
        return status_t(daal::services::ErrorMethodNotSupported);
    }

    void freeDataMemoryImpl() DAAL_C11_OVERRIDE {
        original_table_ = homogen_table{};
    }

    template <typename BlockData>
    status_t readRows(size_t vector_idx, size_t vector_num, rw_mode_t rwflag, block_desc_t<BlockData>& block) {
        if (rwflag != daal::data_management::readOnly) {
            return status_t(daal::services::ErrorMethodNotSupported);
        }
        const std::int64_t row_count = original_table_.get_row_count();
        const std::int64_t column_count = original_table_.get_column_count();
        const block_info info { block, vector_idx, vector_num };

        if (info.row_begin_index >= row_count || info.row_end_index > row_count) {
            return status_t(daal::services::ErrorIncorrectIndex);
        }

        try {
            array<BlockData> values;
            auto block_ptr = block.getBlockPtr();
            if (block_ptr != nullptr && info.size >= info.row_count * column_count) {
                values.reset(block_ptr, info.size, dal::empty_delete<BlockData>());
            }

            row_accessor<const BlockData> acc { original_table_ };
            if(block_ptr != pull_values(acc, values, range{ info.row_begin_index, info.row_end_index })) {
                setBlockData(values, info.row_count, column_count, block);
            }
        } catch (const std::exception&) {
            return status_t(daal::services::UnknownError);
        }

        return status_t();
    }

    template <typename BlockData>
    status_t readColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, rw_mode_t rwflag,
                              block_desc_t<BlockData> & block) {
        if (rwflag != daal::data_management::readOnly) {
            return status_t(daal::services::ErrorMethodNotSupported);
        }
        const std::int64_t row_count = original_table_.get_row_count();
        const std::int64_t column_count = original_table_.get_column_count();

        const block_info info { block, vector_idx, value_num, feature_idx };

        if (info.column_index >= column_count || info.row_begin_index >= row_count || info.row_end_index > row_count) {
            return status_t(daal::services::ErrorIncorrectIndex);
        }

        try {
            array<BlockData> values;
            auto block_ptr = block.getBlockPtr();
            if (block_ptr != nullptr && info.size >= info.row_count) {
                values.reset(block_ptr, info.size, dal::empty_delete<BlockData>());
            }

            column_accessor<const BlockData> acc { original_table_ };
            if(block_ptr != pull_values(acc, values, info.column_index, range{ info.row_begin_index, info.row_end_index })) {
                setBlockData(values, info.row_count, 1, block);
            }
        } catch (const std::exception&) {
            return status_t(daal::services::UnknownError);
        }

        return status_t();
    }

    template <typename BlockData>
    void setBlockData(array<BlockData>& data, std::int64_t row_count, std::int64_t column_count, block_desc_t<BlockData>& block) {
        auto raw_ptr = const_cast<BlockData*>(data.get_data());
        if constexpr (is_host_only) {
            auto data_shared = daal::services::SharedPtr<BlockData>(raw_ptr,
                    [data](auto ptr) mutable { data.reset(); });
            block.setSharedPtr(data_shared, column_count, row_count);
        }
    #ifdef ONEAPI_DAL_DATA_PARALLEL
        else {
            daal::services::Buffer<BlockData> buffer(raw_ptr, row_count*column_count, sycl::usm::alloc::shared);
            block.setBuffer(buffer, column_count, row_count);
        }
    #endif
    }

    template <typename Accessor, typename... Args>
    auto pull_values(const Accessor& acc, Args&&... args) {
        if constexpr (is_host_only) {
            return acc.pull(std::forward<Args>(args)...);
        }
    #ifdef ONEAPI_DAL_DATA_PARALLEL
        else {
            return acc.pull(policy_.get_queue(), std::forward<Args>(args)..., sycl::usm::alloc::shared);
        }
    #endif
    }

    homogen_table_adapter(const Policy& policy, const homogen_table& table, status_t& stat)
        : base(const_cast<Data*>(table.get_data<Data>()),
               table.get_column_count(),
               table.get_row_count()),
          policy_(policy) {

        if (table.has_data() == false ||
            table.get_data_layout() != data_layout::row_major) {
            stat.add(daal::services::ErrorIncorrectParameter);
            return;
        }

        original_table_ = table;

        auto meta = original_table_.get_metadata();
        dtype_ = meta.get_data_type(0);

        this->_memStatus = original_table_.has_data() ?
                            daal::data_management::NumericTableIface::userAllocated :
                            daal::data_management::NumericTableIface::notAllocated;
        this->_layout = daal::data_management::NumericTableIface::aos;
    }

private:
    Policy policy_;
    homogen_table original_table_;
    data_type dtype_;
};

} // namespace oneapi::dal::backend::interop
