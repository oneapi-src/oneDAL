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

class homogen_table_adapter : public daal::data_management::NumericTable {
    using ptr_t = daal::services::SharedPtr<homogen_table_adapter>;
    using status_t = daal::services::Status;
    using rw_mode_t = daal::data_management::ReadWriteMode;

    template <typename T>
    using block_desc_t = daal::data_management::BlockDescriptor<T>;

public:
    static ptr_t create(const homogen_table& table) {
        status_t internal_stat;
        auto result = ptr_t { new homogen_table_adapter(table, internal_stat) };
        status_to_exception(internal_stat);
        return result;
    }

    ~homogen_table_adapter() { freeDataMemoryImpl(); }

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

    template <typename Data>
    status_t readRows(size_t vector_idx, size_t vector_num, rw_mode_t rwflag, block_desc_t<Data>& block) {
        if (rwflag != daal::data_management::readOnly) {
            return status_t(daal::services::ErrorMethodNotSupported);
        }
        const std::int64_t row_count = original_table_.get_row_count();
        const std::int64_t column_count = original_table_.get_column_count();

        if (vector_idx >= row_count || vector_idx+vector_num > row_count) {
            return status_t(daal::services::ErrorIncorrectIndex);
        }

        try {
            row_accessor<const Data> acc { original_table_ };
            auto values = acc.pull({ vector_idx, vector_idx + vector_num });

            auto data_ptr = const_cast<Data*>(values.get_data());
            daal::services::SharedPtr<Data> daal_shared_ptr(data_ptr, [values](auto ptr) mutable { values.reset(); });

            block.setSharedPtr(daal_shared_ptr, column_count, vector_num);
        } catch (const std::exception&) {
            return status_t(daal::services::UnknownError);
        }

        return status_t();
    }

    template <typename Data>
    status_t readColumnValues(size_t feature_idx, size_t vector_idx, size_t value_num, rw_mode_t rwflag,
                              block_desc_t<Data> & block) {
        if (rwflag != daal::data_management::readOnly) {
            return status_t(daal::services::ErrorMethodNotSupported);
        }
        const std::int64_t row_count = original_table_.get_row_count();
        const std::int64_t column_count = original_table_.get_column_count();

        if (feature_idx >= column_count || vector_idx >= row_count || vector_idx+value_num > row_count) {
            return status_t(daal::services::ErrorIncorrectIndex);
        }

        try {
            column_accessor<const Data> acc { original_table_ };
            auto values = acc.pull(feature_idx, { vector_idx, vector_idx + value_num });

            auto data_ptr = const_cast<Data*>(values.get_data());
            daal::services::SharedPtr<Data> daal_shared_ptr(data_ptr, [values](auto ptr) mutable { values.reset(); });

            block.setSharedPtr(daal_shared_ptr, 1, value_num);
        } catch (const std::exception&) {
            return status_t(daal::services::UnknownError);
        }

        return status_t();
    }

    void freeDataMemoryImpl() DAAL_C11_OVERRIDE {
        original_table_ = homogen_table{};
    }

    homogen_table_adapter(const homogen_table& table, status_t& stat)
        : NumericTable(table.get_column_count(), table.get_row_count()) {

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
    homogen_table original_table_;
    data_type dtype_;
};

} // namespace oneapi::dal::backend::interop
