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

#include "oneapi/dal/data/table.hpp"
#include "oneapi/dal/data/table_builder.hpp"
#include "daal/include/data_management/data_source/file_data_source.h"
#include "daal/include/data_management/data_source/csv_feature_manager.h"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

namespace oneapi::dal::backend {

class csv_table_reader_impl {
public:
    csv_table_reader_impl() : delimiter_(','), parse_header_(false) {}

    table read(const char * file_name) {
        using namespace daal::data_management;

        CsvDataSourceOptions csv_options =
            CsvDataSourceOptions::allocateNumericTable |
            CsvDataSourceOptions::createDictionaryFromContext |
            (parse_header_ ? CsvDataSourceOptions::parseHeader : CsvDataSourceOptions::byDefault);

        FileDataSource<CSVFeatureManager> data_source(file_name, csv_options);
        data_source.getFeatureManager().setDelimiter(delimiter_);
        data_source.loadDataBlock();

        return oneapi::dal::backend::interop::convert_from_daal_homogen_table<DAAL_DATA_TYPE>(data_source.getNumericTable());
    }

    void set_parse_header(bool parse_header) {
        parse_header_ = parse_header;
    }

    void set_delimiter(char delimiter) {
        delimiter_ = delimiter;
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    table read(sycl::queue & queue,
               const char * file_name) {

        using namespace daal::data_management;

        CsvDataSourceOptions csv_options =
            CsvDataSourceOptions::allocateNumericTable |
            CsvDataSourceOptions::createDictionaryFromContext |
            (parse_header_ ? CsvDataSourceOptions::parseHeader : CsvDataSourceOptions::byDefault);

        FileDataSource<CSVFeatureManager> data_source(file_name, csv_options);
        data_source.getFeatureManager().setDelimiter(delimiter_);
        data_source.loadDataBlock();

        auto nt = data_source.getNumericTable();

        daal::data_management::BlockDescriptor<DAAL_DATA_TYPE> block;
        const std::int64_t row_count    = nt->getNumberOfRows();
        const std::int64_t column_count = nt->getNumberOfColumns();

        nt->getBlockOfRows(0, row_count, daal::data_management::readOnly, block);
        DAAL_DATA_TYPE* data = block.getBlockPtr();

        auto arr = array<DAAL_DATA_TYPE>::empty(queue, row_count * column_count);
        detail::memcpy(queue, arr.get_mutable_data(), data, sizeof(DAAL_DATA_TYPE) * row_count * column_count);

        nt->releaseBlockOfRows(block);

        return homogen_table_builder{}.reset(arr, row_count, column_count).build();
    }
#endif

private:
    char delimiter_;
    bool parse_header_;
};

} // namespace oneapi::dal::backend
