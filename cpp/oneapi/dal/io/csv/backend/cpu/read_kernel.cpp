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

#include <daal/include/data_management/data_source/file_data_source.h>
#include <daal/include/data_management/data_source/csv_feature_manager.h>

#include "oneapi/dal/io/csv/backend/cpu/read_kernel.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include <iostream>

namespace oneapi::dal::csv::backend {

table read_kernel_cpu<table>::operator()(const dal::backend::context_cpu& ctx,
                                         const data_source_base& data_source,
                                         const read_args<table>& args) const {
    using namespace daal::data_management;

    CsvDataSourceOptions csv_options =
        CsvDataSourceOptions::allocateNumericTable |
        CsvDataSourceOptions::createDictionaryFromContext |
        (data_source.get_parse_header() ? CsvDataSourceOptions::parseHeader : CsvDataSourceOptions::byDefault);

    FileDataSource<CSVFeatureManager> daal_data_source(data_source.get_file_name(), csv_options);
    daal_data_source.getFeatureManager().setDelimiter(data_source.get_delimiter());
    daal_data_source.loadDataBlock();

    return oneapi::dal::backend::interop::convert_from_daal_homogen_table<DAAL_DATA_TYPE>(
        daal_data_source.getNumericTable());
}

} // namespace oneapi::dal::csv::backend
