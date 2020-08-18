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

#ifndef ONEAPI_DAL_DATA_CONVERSION

    #define ONEAPI_DAL_DATA_CONVERSION
    #include "daal/include/data_management/data_source/csv_feature_manager.h"
    #include "daal/include/data_management/data_source/file_data_source.h"
    #undef ONEAPI_DAL_DATA_CONVERSION

#endif

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"
#include "oneapi/dal/detail/memory.hpp"
#include "oneapi/dal/io/csv/backend/gpu/read_kernel.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::csv::backend {

table read_kernel_gpu<table>::operator()(const dal::backend::context_gpu& ctx,
                                         const data_source_base& data_source,
                                         const read_args<table>& args) const {
    auto& queue = ctx.get_queue();

    using namespace daal::data_management;

    CsvDataSourceOptions csv_options =
        CsvDataSourceOptions::allocateNumericTable |
        CsvDataSourceOptions::createDictionaryFromContext |
        (data_source.get_parse_header() ? CsvDataSourceOptions::parseHeader
                                        : CsvDataSourceOptions::byDefault);

    FileDataSource<CSVFeatureManager> daal_data_source(data_source.get_file_name(), csv_options);
    daal_data_source.getFeatureManager().setDelimiter(data_source.get_delimiter());
    daal_data_source.loadDataBlock();

    auto nt = daal_data_source.getNumericTable();

    daal::data_management::BlockDescriptor<DAAL_DATA_TYPE> block;
    const std::int64_t row_count    = nt->getNumberOfRows();
    const std::int64_t column_count = nt->getNumberOfColumns();

    nt->getBlockOfRows(0, row_count, daal::data_management::readOnly, block);
    DAAL_DATA_TYPE* data = block.getBlockPtr();

    auto arr = array<DAAL_DATA_TYPE>::empty(queue, row_count * column_count);
    dal::detail::memcpy(queue,
                        arr.get_mutable_data(),
                        data,
                        sizeof(DAAL_DATA_TYPE) * row_count * column_count);

    nt->releaseBlockOfRows(block);

    return dal::detail::homogen_table_builder{}.reset(arr, row_count, column_count).build();
}

} // namespace oneapi::dal::csv::backend
