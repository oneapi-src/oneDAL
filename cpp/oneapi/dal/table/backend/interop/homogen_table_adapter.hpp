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
#include "oneapi/dal/backend/interop/daal_object_owner.hpp"
#include "oneapi/dal/table/backend/interop/block_info.hpp"

namespace oneapi::dal::backend::interop {

// This class shall be used only to represent immutable data on DAAL side. Any
// attempts to change the data inside objects of that class lead to undefined
// behavior.
template <typename Data>
class homogen_table_adapter : public daal::data_management::HomogenNumericTable<Data> {
    using base = daal::data_management::HomogenNumericTable<Data>;
    using status_t = daal::services::Status;
    using rw_mode_t = daal::data_management::ReadWriteMode;
    using ptr_data_t = daal::services::SharedPtr<Data>;

    template <typename T>
    using block_desc_t = daal::data_management::BlockDescriptor<T>;

protected:
    homogen_table_adapter(const homogen_table& table, status_t& stat);

    bool check_row_indexes_in_range(const block_info& info) const {
        const std::int64_t row_count = original_table_.get_row_count();
        return info.row_begin_index < row_count && info.row_end_index <= row_count;
    }

    bool check_column_index_in_range(const block_info& info) const {
        const std::int64_t column_count = original_table_.get_column_count();
        return info.single_column_requested && info.column_index < column_count;
    }

    const homogen_table& get_original_table() const {
        return original_table_;
    }

    template <typename Body>
    daal::services::Status convert_exception_to_status(Body&& body) {
        try {
            return body();
        }
        catch (const bad_alloc&) {
            return daal::services::ErrorMemoryAllocationFailed;
        }
        catch (const out_of_range&) {
            return daal::services::ErrorIncorrectDataRange;
        }
        catch (...) {
            return daal::services::UnknownError;
        }
    }

private:
    status_t releaseBlockOfRows(block_desc_t<double>& block) override;
    status_t releaseBlockOfRows(block_desc_t<float>& block) override;
    status_t releaseBlockOfRows(block_desc_t<int>& block) override;

    status_t releaseBlockOfColumnValues(block_desc_t<double>& block) override;
    status_t releaseBlockOfColumnValues(block_desc_t<float>& block) override;
    status_t releaseBlockOfColumnValues(block_desc_t<int>& block) override;

    status_t assign(float value) override;
    status_t assign(double value) override;
    status_t assign(int value) override;

    int getSerializationTag() const override;
    status_t serializeImpl(daal::data_management::InputDataArchive* arch) override;
    status_t deserializeImpl(const daal::data_management::OutputDataArchive* arch) override;

    status_t allocateDataMemoryImpl(daal::MemType) override;
    status_t setNumberOfColumnsImpl(std::size_t) override;
    void freeDataMemoryImpl() override;

    homogen_table original_table_;
};

} // namespace oneapi::dal::backend::interop
