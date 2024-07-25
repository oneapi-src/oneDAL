/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/column_accessor.hpp"
#include "oneapi/dal/table/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/table/backend/interop/buffer_adapter.hpp"
#include "oneapi/dal/table/backend/interop/host_heterogen_to_soa_table_adapter.hpp"

namespace oneapi::dal::backend::interop {

namespace daal_dm = daal::data_management;

auto host_heterogen_table_adapter::create(const heterogen_table& table) -> ptr_t {
    status_t status;
    auto* raw = new host_heterogen_table_adapter(table, status);
    interop::status_to_exception(status);
    return ptr_t{ raw };
}

template <typename Integral>
inline auto to_size_t(Integral integral) {
    using dal::detail::integral_cast;
    return integral_cast<std::size_t>(integral);
}

host_heterogen_table_adapter::host_heterogen_table_adapter(const heterogen_table& table,
                                                           status_t& status)
        : base_t{ convert(table.get_metadata()), to_size_t(table.get_row_count()) },
          original_table_(table) {}

auto host_heterogen_table_adapter::getBlockOfRows(std::size_t vector_idx,
                                                  std::size_t vector_num,
                                                  rw_mode_t rwflag,
                                                  block_desc_t<double>& block) -> status_t {
    return convert_exception_to_status([&]() -> status_t {
        return read_rows_impl(vector_idx, vector_num, rwflag, block);
    });
}

auto host_heterogen_table_adapter::getBlockOfRows(std::size_t vector_idx,
                                                  std::size_t vector_num,
                                                  rw_mode_t rwflag,
                                                  block_desc_t<float>& block) -> status_t {
    return convert_exception_to_status([&]() -> status_t {
        return read_rows_impl(vector_idx, vector_num, rwflag, block);
    });
}

auto host_heterogen_table_adapter::getBlockOfRows(std::size_t vector_idx,
                                                  std::size_t vector_num,
                                                  rw_mode_t rwflag,
                                                  block_desc_t<int>& block) -> status_t {
    return convert_exception_to_status([&]() -> status_t {
        return read_rows_impl(vector_idx, vector_num, rwflag, block);
    });
}

auto host_heterogen_table_adapter::getBlockOfColumnValues(std::size_t feature_idx,
                                                          std::size_t vector_idx,
                                                          std::size_t value_num,
                                                          rw_mode_t rwflag,
                                                          block_desc_t<double>& block) -> status_t {
    return convert_exception_to_status([&]() -> status_t {
        return read_column_values_impl(feature_idx, vector_idx, value_num, rwflag, block);
    });
}

auto host_heterogen_table_adapter::getBlockOfColumnValues(std::size_t feature_idx,
                                                          std::size_t vector_idx,
                                                          std::size_t value_num,
                                                          rw_mode_t rwflag,
                                                          block_desc_t<float>& block) -> status_t {
    return convert_exception_to_status([&]() -> status_t {
        return read_column_values_impl(feature_idx, vector_idx, value_num, rwflag, block);
    });
}

auto host_heterogen_table_adapter::getBlockOfColumnValues(std::size_t feature_idx,
                                                          std::size_t vector_idx,
                                                          std::size_t value_num,
                                                          rw_mode_t rwflag,
                                                          block_desc_t<int>& block) -> status_t {
    return convert_exception_to_status([&]() -> status_t {
        return read_column_values_impl(feature_idx, vector_idx, value_num, rwflag, block);
    });
}

auto host_heterogen_table_adapter::releaseBlockOfRows(block_desc_t<double>& block) -> status_t {
    block.reset();
    return status_t();
}

auto host_heterogen_table_adapter::releaseBlockOfRows(block_desc_t<float>& block) -> status_t {
    block.reset();
    return status_t();
}

auto host_heterogen_table_adapter::releaseBlockOfRows(block_desc_t<int>& block) -> status_t {
    block.reset();
    return status_t();
}

auto host_heterogen_table_adapter::releaseBlockOfColumnValues(block_desc_t<double>& block)
    -> status_t {
    block.reset();
    return status_t();
}

auto host_heterogen_table_adapter::releaseBlockOfColumnValues(block_desc_t<float>& block)
    -> status_t {
    block.reset();
    return status_t();
}

auto host_heterogen_table_adapter::releaseBlockOfColumnValues(block_desc_t<int>& block)
    -> status_t {
    block.reset();
    return status_t();
}

auto host_heterogen_table_adapter::assign(float) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

auto host_heterogen_table_adapter::assign(double) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

auto host_heterogen_table_adapter::assign(int) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

auto host_heterogen_table_adapter::allocateDataMemoryImpl(daal::MemType) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

auto host_heterogen_table_adapter::setNumberOfColumnsImpl(std::size_t) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

int host_heterogen_table_adapter::getSerializationTag() const {
    ONEDAL_ASSERT(!"host_soa_table_adapter: getSerializationTag() is not implemented");
    return -1;
}

auto host_heterogen_table_adapter::serializeImpl(daal_dm::InputDataArchive* arch) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

auto host_heterogen_table_adapter::deserializeImpl(const daal_dm::OutputDataArchive* arch)
    -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

void host_heterogen_table_adapter::freeDataMemoryImpl() {
    base_t::freeDataMemoryImpl();
    original_table_ = heterogen_table{};
}

template <typename Data>
auto host_heterogen_table_adapter::read_rows_impl(std::size_t vector_idx,
                                                  std::size_t vector_num,
                                                  rw_mode_t rwflag,
                                                  block_desc_t<Data>& block) -> status_t {
    if (rwflag != daal_dm::readOnly) {
        ONEDAL_ASSERT(!"Data is accessible in read-only mode by design");
        return daal::services::ErrorMethodNotImplemented;
    }

    using dal::detail::integral_cast;
    using dal::detail::check_sum_overflow;

    const auto column_count = original_table_.get_column_count();
    const auto col_count = integral_cast<std::size_t>(column_count);

    const auto row_count = integral_cast<std::int64_t>(vector_num);
    const auto first = integral_cast<std::int64_t>(vector_idx);
    const auto last = check_sum_overflow(first, row_count);

    row_accessor<const Data> accessor{ original_table_ };
    dal::array<Data> data = accessor.pull({ first, last });
    auto [buffer, status] = convert_with_status(data);

    block.setBuffer(buffer, col_count, vector_num);

    return status;
}

template <typename Data>
auto host_heterogen_table_adapter::read_column_values_impl(std::size_t feature_idx,
                                                           std::size_t vector_idx,
                                                           std::size_t value_num,
                                                           rw_mode_t rwflag,
                                                           block_desc_t<Data>& block) -> status_t {
    if (rwflag != daal_dm::readOnly) {
        ONEDAL_ASSERT(!"Data is accessible in read-only mode by design");
        return daal::services::ErrorMethodNotImplemented;
    }

    using dal::detail::integral_cast;
    using dal::detail::check_sum_overflow;

    const auto column_idx = integral_cast<std::int64_t>(feature_idx);
    const auto row_count = integral_cast<std::int64_t>(value_num);
    const auto first = integral_cast<std::int64_t>(vector_idx);
    const auto last = check_sum_overflow(first, row_count);

    column_accessor<const Data> accessor{ original_table_ };
    auto data = accessor.pull(column_idx, { first, last });
    auto [buffer, status] = convert_with_status(data);

    block.setBuffer(buffer, 1ul, value_num);

    return status;
}

} // namespace oneapi::dal::backend::interop
