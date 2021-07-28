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

#pragma once

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/backend/common_kernels.hpp"
#include "oneapi/dal/table/backend/csr_kernels.hpp"
#include "oneapi/dal/table/detail/csr_access_iface.hpp"

namespace oneapi::dal::backend {

class csr_table_impl : public detail::csr_table_template<csr_table_impl> {
public:
    csr_table_impl()
            : col_count_(0),
              row_count_(0),
              layout_(data_layout::row_major),
              csr_indexing_(detail::csr_indexing::one_based) {}

    csr_table_impl(std::int64_t column_count,
                   std::int64_t row_count,
                   const array<byte_t>& data,
                   const array<std::int64_t>& column_indices,
                   const array<std::int64_t>& row_indices,
                   data_type dtype,
                   detail::csr_indexing indexing)
            : meta_(create_metadata(column_count, dtype)),
              data_(data),
              column_indices_(column_indices),
              row_indices_(row_indices),
              col_count_(column_count),
              row_count_(row_count),
              layout_(data_layout::row_major),
              csr_indexing_(indexing) {
        using error_msg = dal::detail::error_messages;

        if (row_count <= 0) {
            throw dal::domain_error(error_msg::rc_leq_zero());
        }

        if (column_count <= 0) {
            throw dal::domain_error(error_msg::cc_leq_zero());
        }

        if (indexing != detail::csr_indexing::one_based) {
            throw dal::domain_error(detail::error_messages::zero_based_indexing_is_not_supported());
        }

        const int64_t element_count = row_indices_[row_count] - 1;
        const int64_t dtype_size = detail::get_data_type_size(dtype);

        detail::check_mul_overflow(element_count, dtype_size);
        if (data.get_count() != element_count * dtype_size) {
            throw dal::domain_error(error_msg::invalid_data_block_size());
        }
    }

    // Needs to be overriden for backward compatibility. Should be remove in oneDAL 2022.1.
    detail::access_iface_host& get_access_iface_host() const override {
        using msg = detail::error_messages;
        throw dal::internal_error{ msg::object_does_not_provide_access_to_rows_or_columns() };
    }

#ifdef ONEDAL_DATA_PARALLEL
    // Needs to be overriden for backward compatibility. Should be remove in oneDAL 2022.1.
    detail::access_iface_dpc& get_access_iface_dpc() const override {
        using msg = detail::error_messages;
        throw dal::internal_error{ msg::object_does_not_provide_access_to_rows_or_columns() };
    }
#endif

    std::int64_t get_column_count() const override {
        return col_count_;
    }

    std::int64_t get_row_count() const override {
        return row_count_;
    }

    const table_metadata& get_metadata() const override {
        return meta_;
    }

    std::int64_t get_kind() const override {
        return 10;
    }

    array<byte_t> get_data() const override {
        return data_;
    }

    array<std::int64_t> get_column_indices() const override {
        return column_indices_;
    }

    array<std::int64_t> get_row_indices() const override {
        return row_indices_;
    }

    data_layout get_data_layout() const override {
        return layout_;
    }

    template <typename T>
    void pull_csr_block_template(const detail::default_host_policy& policy,
                                 detail::csr_block<T>& block,
                                 const detail::csr_indexing& indexing,
                                 const range& rows) const {
        csr_info origin_info{ meta_.get_data_type(0),
                              layout_,
                              row_count_,
                              col_count_,
                              row_indices_[row_count_] - row_indices_[0],
                              csr_indexing_ };

        // Overflow is checked here
        check_block_row_range(rows);

        if (indexing != detail::csr_indexing::one_based) {
            throw dal::unimplemented(
                detail::error_messages::zero_based_indexing_is_not_supported());
        }

        block_info block_info{ rows.start_idx, rows.get_element_count(row_count_), indexing };

        csr_pull_block(policy,
                       origin_info,
                       block_info,
                       data_,
                       column_indices_,
                       row_indices_,
                       block,
                       alloc_kind::host);
    }

private:
    void check_block_row_range(const range& rows) const {
        const std::int64_t range_row_count = rows.get_element_count(row_count_);
        detail::check_sum_overflow(rows.start_idx, range_row_count);
        if (rows.start_idx + range_row_count > row_count_) {
            throw range_error{ detail::error_messages::invalid_range_of_rows() };
        }
    }

    table_metadata meta_;
    array<byte_t> data_;
    array<std::int64_t> column_indices_;
    array<std::int64_t> row_indices_;
    std::int64_t col_count_;
    std::int64_t row_count_;
    data_layout layout_;
    detail::csr_indexing csr_indexing_;
};

} // namespace oneapi::dal::backend
