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

#include "oneapi/dal/table/backend/interop/usm_homogen_table_adapter.hpp"

namespace oneapi::dal::backend::interop {

namespace daal_dm = daal::data_management;

template <typename Data>
auto usm_homogen_table_adapter<Data>::create(sycl::queue& q, const homogen_table& table) -> ptr_t {
    status_t internal_stat;
    auto result = ptr_t{ new usm_homogen_table_adapter(q, table, internal_stat) };
    status_to_exception(internal_stat);
    return result;
}

template <typename Data>
usm_homogen_table_adapter<Data>::usm_homogen_table_adapter(sycl::queue& q,
                                                           const homogen_table& table,
                                                           status_t& stat)
        : base(table, stat),
          queue_(q) {}

template <typename Data>
auto usm_homogen_table_adapter<Data>::getBlockOfRows(std::size_t vector_idx,
                                                     std::size_t vector_num,
                                                     rw_mode_t rwflag,
                                                     block_desc_t<double>& block) -> status_t {
    return base::convert_exception_to_status([&]() {
        return read_rows_impl(vector_idx, vector_num, rwflag, block);
    });
}

template <typename Data>
auto usm_homogen_table_adapter<Data>::getBlockOfRows(std::size_t vector_idx,
                                                     std::size_t vector_num,
                                                     rw_mode_t rwflag,
                                                     block_desc_t<float>& block) -> status_t {
    return base::convert_exception_to_status([&]() {
        return read_rows_impl(vector_idx, vector_num, rwflag, block);
    });
}

template <typename Data>
auto usm_homogen_table_adapter<Data>::getBlockOfRows(std::size_t vector_idx,
                                                     std::size_t vector_num,
                                                     rw_mode_t rwflag,
                                                     block_desc_t<int>& block) -> status_t {
    return base::convert_exception_to_status([&]() {
        return read_rows_impl(vector_idx, vector_num, rwflag, block);
    });
}

template <typename Data>
auto usm_homogen_table_adapter<Data>::getBlockOfColumnValues(std::size_t feature_idx,
                                                             std::size_t vector_idx,
                                                             std::size_t value_num,
                                                             rw_mode_t rwflag,
                                                             block_desc_t<double>& block)
    -> status_t {
    return base::convert_exception_to_status([&]() {
        return read_column_values_impl(feature_idx, vector_idx, value_num, rwflag, block);
    });
}

template <typename Data>
auto usm_homogen_table_adapter<Data>::getBlockOfColumnValues(std::size_t feature_idx,
                                                             std::size_t vector_idx,
                                                             std::size_t value_num,
                                                             rw_mode_t rwflag,
                                                             block_desc_t<float>& block)
    -> status_t {
    return base::convert_exception_to_status([&]() {
        return read_column_values_impl(feature_idx, vector_idx, value_num, rwflag, block);
    });
}

template <typename Data>
auto usm_homogen_table_adapter<Data>::getBlockOfColumnValues(std::size_t feature_idx,
                                                             std::size_t vector_idx,
                                                             std::size_t value_num,
                                                             rw_mode_t rwflag,
                                                             block_desc_t<int>& block) -> status_t {
    return base::convert_exception_to_status([&]() {
        return read_column_values_impl(feature_idx, vector_idx, value_num, rwflag, block);
    });
}

template <typename Data>
template <typename BlockData>
auto usm_homogen_table_adapter<Data>::read_rows_impl(std::size_t vector_idx,
                                                     std::size_t vector_num,
                                                     rw_mode_t rwflag,
                                                     block_desc_t<BlockData>& block) -> status_t {
    if (rwflag != daal_dm::readOnly) {
        ONEDAL_ASSERT(!"Data is accessible in read-only mode by design");
        return daal::services::ErrorMethodNotImplemented;
    }

    const block_info info{ block, vector_idx, vector_num };
    if (!base::check_row_indexes_in_range(info)) {
        return daal::services::ErrorIncorrectIndex;
    }

    const auto [buffer, status] = pull_rows_buffer<BlockData>(info);
    if (status.ok()) {
        block.setDetails(0, vector_idx, rwflag);
        block.setBuffer(buffer, info.row_count, base::get_original_table().get_column_count());
    }

    return status;
}

template <typename Data>
template <typename BlockData>
auto usm_homogen_table_adapter<Data>::read_column_values_impl(std::size_t feature_idx,
                                                              std::size_t vector_idx,
                                                              std::size_t value_num,
                                                              rw_mode_t rwflag,
                                                              block_desc_t<BlockData>& block)
    -> status_t {
    if (rwflag != daal_dm::readOnly) {
        ONEDAL_ASSERT(!"Data is accessible in read-only mode by design");
        return daal::services::ErrorMethodNotImplemented;
    }

    const block_info info{ block, vector_idx, value_num, feature_idx };
    if (!base::check_row_indexes_in_range(info) || !base::check_column_index_in_range(info)) {
        return daal::services::ErrorIncorrectIndex;
    }

    const auto [buffer, status] = pull_rows_buffer<BlockData>(info);
    if (status.ok()) {
        block.setDetails(feature_idx, vector_idx, rwflag);
        block.setBuffer(buffer, info.row_count, base::get_original_table().get_column_count());
    }

    return status;
}

template <typename Data>
template <typename BlockData>
auto usm_homogen_table_adapter<Data>::convert_to_daal_buffer(const array<BlockData>& ary) const
    -> daal_buffer_and_status_t<BlockData> {
    status_t status;
    // `const_cast` is safe assuming read-only access to the table on DAAL side and
    // correct `rwflag` passed to `getBlockOfRows` or `getBlockOfColumnValues`.
    const auto buffer =
        daal_buffer_t<BlockData>{ const_cast<BlockData*>(ary.get_data()),
                                  dal::detail::integral_cast<std::size_t>(ary.get_count()),
                                  queue_,
                                  status };
    return { buffer, status };
}

constexpr inline sycl::usm::alloc get_accessor_alloc_kind() {
    // We always request device-allocated data assuming adapter is used within
    // DAAL kernels, which rely on device USM.
    return sycl::usm::alloc::device;
}

template <typename Data>
template <typename BlockData>
auto usm_homogen_table_adapter<Data>::pull_rows_buffer(const block_info& info)
    -> daal_buffer_and_status_t<BlockData> {
    const auto values = //
        row_accessor<const BlockData>{ base::get_original_table() } //
            .pull(queue_, info.get_row_range(), get_accessor_alloc_kind());
    return convert_to_daal_buffer(values);
}

template <typename Data>
template <typename BlockData>
auto usm_homogen_table_adapter<Data>::pull_columns_buffer(const block_info& info)
    -> daal_buffer_and_status_t<BlockData> {
    const auto values = //
        column_accessor<const BlockData>{ base::get_original_table() } //
            .pull(queue_, info.column_index, info.get_row_range(), get_accessor_alloc_kind());
    return convert_to_daal_buffer(values);
}

template class usm_homogen_table_adapter<std::int32_t>;
template class usm_homogen_table_adapter<float>;
template class usm_homogen_table_adapter<double>;

} // namespace oneapi::dal::backend::interop
