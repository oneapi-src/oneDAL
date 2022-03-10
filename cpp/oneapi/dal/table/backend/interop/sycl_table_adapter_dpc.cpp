/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/table/backend/interop/sycl_table_adapter.hpp"

namespace oneapi::dal::backend::interop {

namespace daal_dm = daal::data_management;

template <typename Body>
static daal::services::Status convert_exception_to_status(Body&& body) {
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

static daal_dm::features::FeatureType get_daal_feature_type(feature_type t) {
    switch (t) {
        case feature_type::nominal: return daal_dm::features::DAAL_CATEGORICAL;
        case feature_type::ordinal: return daal_dm::features::DAAL_ORDINAL;
        case feature_type::interval: return daal_dm::features::DAAL_CONTINUOUS;
        case feature_type::ratio: return daal_dm::features::DAAL_CONTINUOUS;
        default: throw dal::internal_error(detail::error_messages::unsupported_feature_type());
    }
}

static void convert_feature_information_to_daal(const table_metadata& src,
                                                daal_dm::NumericTableDictionary& dst) {
    ONEDAL_ASSERT(std::size_t(src.get_feature_count()) == dst.getNumberOfFeatures());
    for (std::int64_t i = 0; i < src.get_feature_count(); i++) {
        auto& daal_feature = dst[i];
        daal_feature.featureType = get_daal_feature_type(src.get_feature_type(i));
    }
}

auto sycl_table_adapter::create(const sycl::queue& q, const table& table) -> ptr_t {
    status_t internal_stat;
    auto result = ptr_t{ new sycl_table_adapter(q, table, internal_stat) };
    status_to_exception(internal_stat);
    return result;
}

sycl_table_adapter::sycl_table_adapter(const sycl::queue& q, const table& table, status_t& status)
        : base(table.get_column_count(), table.get_row_count(), daal_dm::DictionaryIface::equal),
          queue_(q),
          original_table_(table) {
    if (!status.ok()) {
        return;
    }
    else if (!table.has_data()) {
        status.add(daal::services::ErrorIncorrectParameter);
        return;
    }

    this->_memStatus = daal_dm::NumericTableIface::userAllocated;
    this->_layout = daal_dm::NumericTableIface::aos;

    convert_feature_information_to_daal(original_table_.get_metadata(),
                                        *this->getDictionarySharedPtr());
}

auto sycl_table_adapter::getBlockOfRows(std::size_t vector_idx,
                                        std::size_t vector_num,
                                        rw_mode_t rwflag,
                                        block_desc_t<double>& block) -> status_t {
    return convert_exception_to_status([&]() {
        return read_rows_impl(vector_idx, vector_num, rwflag, block);
    });
}

auto sycl_table_adapter::getBlockOfRows(std::size_t vector_idx,
                                        std::size_t vector_num,
                                        rw_mode_t rwflag,
                                        block_desc_t<float>& block) -> status_t {
    return convert_exception_to_status([&]() {
        return read_rows_impl(vector_idx, vector_num, rwflag, block);
    });
}

auto sycl_table_adapter::getBlockOfRows(std::size_t vector_idx,
                                        std::size_t vector_num,
                                        rw_mode_t rwflag,
                                        block_desc_t<int>& block) -> status_t {
    return convert_exception_to_status([&]() {
        return read_rows_impl(vector_idx, vector_num, rwflag, block);
    });
}

auto sycl_table_adapter::getBlockOfColumnValues(std::size_t feature_idx,
                                                std::size_t vector_idx,
                                                std::size_t value_num,
                                                rw_mode_t rwflag,
                                                block_desc_t<double>& block) -> status_t {
    return convert_exception_to_status([&]() {
        return read_column_values_impl(feature_idx, vector_idx, value_num, rwflag, block);
    });
}

auto sycl_table_adapter::getBlockOfColumnValues(std::size_t feature_idx,
                                                std::size_t vector_idx,
                                                std::size_t value_num,
                                                rw_mode_t rwflag,
                                                block_desc_t<float>& block) -> status_t {
    return convert_exception_to_status([&]() {
        return read_column_values_impl(feature_idx, vector_idx, value_num, rwflag, block);
    });
}

auto sycl_table_adapter::getBlockOfColumnValues(std::size_t feature_idx,
                                                std::size_t vector_idx,
                                                std::size_t value_num,
                                                rw_mode_t rwflag,
                                                block_desc_t<int>& block) -> status_t {
    return convert_exception_to_status([&]() {
        return read_column_values_impl(feature_idx, vector_idx, value_num, rwflag, block);
    });
}

auto sycl_table_adapter::releaseBlockOfRows(block_desc_t<double>& block) -> status_t {
    block.reset();
    return status_t();
}

auto sycl_table_adapter::releaseBlockOfRows(block_desc_t<float>& block) -> status_t {
    block.reset();
    return status_t();
}

auto sycl_table_adapter::releaseBlockOfRows(block_desc_t<int>& block) -> status_t {
    block.reset();
    return status_t();
}

auto sycl_table_adapter::releaseBlockOfColumnValues(block_desc_t<double>& block) -> status_t {
    block.reset();
    return status_t();
}

auto sycl_table_adapter::releaseBlockOfColumnValues(block_desc_t<float>& block) -> status_t {
    block.reset();
    return status_t();
}

auto sycl_table_adapter::releaseBlockOfColumnValues(block_desc_t<int>& block) -> status_t {
    block.reset();
    return status_t();
}

auto sycl_table_adapter::assign(float) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

auto sycl_table_adapter::assign(double) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

auto sycl_table_adapter::assign(int) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

auto sycl_table_adapter::allocateDataMemoryImpl(daal::MemType) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

auto sycl_table_adapter::setNumberOfColumnsImpl(std::size_t) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

int sycl_table_adapter::getSerializationTag() const {
    ONEDAL_ASSERT(!"sycl_table_adapter: getSerializationTag() is not implemented");
    return -1;
}

auto sycl_table_adapter::serializeImpl(daal_dm::InputDataArchive* arch) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

auto sycl_table_adapter::deserializeImpl(const daal_dm::OutputDataArchive* arch) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

void sycl_table_adapter::freeDataMemoryImpl() {
    base::freeDataMemoryImpl();
    original_table_ = homogen_table{};
}

template <typename BlockData>
auto sycl_table_adapter::read_rows_impl(std::size_t vector_idx,
                                        std::size_t vector_num,
                                        rw_mode_t rwflag,
                                        block_desc_t<BlockData>& block) -> status_t {
    if (rwflag != daal_dm::readOnly) {
        ONEDAL_ASSERT(!"Data is accessible in read-only mode by design");
        return daal::services::ErrorMethodNotImplemented;
    }

    const block_info info{ block, vector_idx, vector_num };
    if (!check_row_indexes_in_range(info)) {
        return daal::services::ErrorIncorrectIndex;
    }

    const auto [buffer, status] = pull_rows_buffer<BlockData>(info);
    if (status.ok()) {
        block.setDetails(0, vector_idx, rwflag);
        block.setBuffer(buffer, info.row_count, original_table_.get_column_count());
    }

    return status;
}

template <typename BlockData>
auto sycl_table_adapter::read_column_values_impl(std::size_t feature_idx,
                                                 std::size_t vector_idx,
                                                 std::size_t value_num,
                                                 rw_mode_t rwflag,
                                                 block_desc_t<BlockData>& block) -> status_t {
    if (rwflag != daal_dm::readOnly) {
        ONEDAL_ASSERT(!"Data is accessible in read-only mode by design");
        return daal::services::ErrorMethodNotImplemented;
    }

    const block_info info{ block, vector_idx, value_num, feature_idx };
    if (!check_row_indexes_in_range(info) || !check_column_index_in_range(info)) {
        return daal::services::ErrorIncorrectIndex;
    }

    const auto [buffer, status] = pull_columns_buffer<BlockData>(info);
    if (status.ok()) {
        block.setDetails(feature_idx, vector_idx, rwflag);
        block.setBuffer(buffer, info.row_count, original_table_.get_column_count());
    }

    return status;
}

template <typename BlockData>
auto sycl_table_adapter::convert_to_daal_buffer(const array<BlockData>& ary) const
    -> daal_buffer_and_status_t<BlockData> {
    using daal::services::SharedPtr;

    status_t status;
    ONEDAL_ASSERT(ary.get_data() != nullptr);

    // `const_cast` is safe assuming read-only access to the table on DAAL side and
    // correct `rwflag` passed to `getBlockOfRows` or `getBlockOfColumnValues`.
    SharedPtr<BlockData> ary_data_shared(const_cast<BlockData*>(ary.get_data()),
                                         daal_object_owner{ ary });

    const auto buffer =
        daal_buffer_t<BlockData>{ std::move(ary_data_shared),
                                  dal::detail::integral_cast<std::size_t>(ary.get_count()),
                                  queue_,
                                  status };
    return { buffer, status };
}

bool sycl_table_adapter::check_row_indexes_in_range(const block_info& info) const {
    const std::int64_t row_count = original_table_.get_row_count();
    return info.row_begin_index < row_count && info.row_end_index <= row_count;
}

bool sycl_table_adapter::check_column_index_in_range(const block_info& info) const {
    const std::int64_t column_count = original_table_.get_column_count();
    return info.single_column_requested && info.column_index < column_count;
}

constexpr inline sycl::usm::alloc get_accessor_alloc_kind() {
    // We always request device-allocated data assuming adapter is used within
    // DAAL kernels, which rely on device USM.
    return sycl::usm::alloc::device;
}

template <typename BlockData>
auto sycl_table_adapter::pull_rows_buffer(const block_info& info)
    -> daal_buffer_and_status_t<BlockData> {
    const auto values = //
        row_accessor<const BlockData>{ original_table_ } //
            .pull(queue_, info.get_row_range(), get_accessor_alloc_kind());
    return convert_to_daal_buffer(values);
}

template <typename BlockData>
auto sycl_table_adapter::pull_columns_buffer(const block_info& info)
    -> daal_buffer_and_status_t<BlockData> {
    const auto values = //
        column_accessor<const BlockData>{ original_table_ } //
            .pull(queue_, info.column_index, info.get_row_range(), get_accessor_alloc_kind());
    return convert_to_daal_buffer(values);
}

} // namespace oneapi::dal::backend::interop
