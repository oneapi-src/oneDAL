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

#include <data_management/data_source/csv_feature_manager.h>
#include <data_management/data_source/file_data_source.h>

#include "oneapi/dal/detail/memory.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/io/csv/read_types.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::csv::detail {

template <typename table, typename Context, typename... Options>
struct ONEAPI_DAL_EXPORT read_ops_dispatcher {
    table operator()(const Context&, const data_source_base&, const read_args<table>&) const;
};

template <>
table read_ops_dispatcher<table, dal::detail::host_policy>::operator()(
    const dal::detail::host_policy& ctx,
    const data_source_base& data_source,
    const read_args<table>& args) const {
    using namespace daal::data_management;

    CsvDataSourceOptions csv_options =
        CsvDataSourceOptions::allocateNumericTable |
        CsvDataSourceOptions::createDictionaryFromContext |
        (data_source.get_parse_header() ? CsvDataSourceOptions::parseHeader
                                        : CsvDataSourceOptions::byDefault);

    FileDataSource<CSVFeatureManager> daal_data_source(data_source.get_file_name(), csv_options);
    daal_data_source.getFeatureManager().setDelimiter(data_source.get_delimiter());
    daal_data_source.loadDataBlock();

    const auto daal_table = daal_data_source.getNumericTable();

    daal::data_management::BlockDescriptor<DAAL_DATA_TYPE> block;
    const std::int64_t row_count    = daal_table->getNumberOfRows();
    const std::int64_t column_count = daal_table->getNumberOfColumns();

    daal_table->getBlockOfRows(0, row_count, readOnly, block);
    DAAL_DATA_TYPE* data = block.getBlockPtr();
    array<DAAL_DATA_TYPE> arr(data,
                              row_count * column_count,
                              [daal_table, block](DAAL_DATA_TYPE* p) mutable {
                                  daal_table->releaseBlockOfRows(block);
                              });

    return dal::detail::homogen_table_builder{}.reset(arr, row_count, column_count).build();
}

template struct ONEAPI_DAL_EXPORT read_ops_dispatcher<table, dal::detail::host_policy>;

#ifdef ONEAPI_DAL_DATA_PARALLEL

template <>
table read_ops_dispatcher<table, dal::detail::data_parallel_policy>::operator()(
    const dal::detail::data_parallel_policy& ctx,
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

    auto daal_table = daal_data_source.getNumericTable();

    daal::data_management::BlockDescriptor<DAAL_DATA_TYPE> block;
    const std::int64_t row_count    = daal_table->getNumberOfRows();
    const std::int64_t column_count = daal_table->getNumberOfColumns();

    daal_table->getBlockOfRows(0, row_count, daal::data_management::readOnly, block);
    DAAL_DATA_TYPE* data = block.getBlockPtr();

    auto arr = array<DAAL_DATA_TYPE>::empty(queue, row_count * column_count);
    dal::detail::memcpy(queue,
                        arr.get_mutable_data(),
                        data,
                        sizeof(DAAL_DATA_TYPE) * row_count * column_count);

    daal_table->releaseBlockOfRows(block);

    return dal::detail::homogen_table_builder{}.reset(arr, row_count, column_count).build();
}

template struct ONEAPI_DAL_EXPORT read_ops_dispatcher<table, dal::detail::data_parallel_policy>;

#endif

template <typename table, typename Descriptor>
struct read_ops {
    using input_t           = read_args<table>;
    using result_t          = table;
    using descriptor_base_t = data_source_base;

    void check_preconditions(const Descriptor& data_source, const input_t& input) const {}

    void check_postconditions(const Descriptor& data_source,
                              const input_t& input,
                              const result_t& result) const {}

    template <typename Context>
    auto operator()(const Context& ctx,
                    const Descriptor& desc,
                    const read_args<table>& args) const {
        check_preconditions(desc, args);
        const auto result = read_ops_dispatcher<table, Context>()(ctx, desc, args);
        check_postconditions(desc, args, result);
        return result;
    }
};

} // namespace oneapi::dal::csv::detail
