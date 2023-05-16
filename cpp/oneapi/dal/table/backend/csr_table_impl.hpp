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
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/serialization.hpp"

#include <iostream>

namespace oneapi::dal::backend {

class csr_table_impl : public detail::csr_table_template<csr_table_impl>,
                       public ONEDAL_SERIALIZABLE(csr_table_id) {
public:
    csr_table_impl()
            : col_count_(0),
              row_count_(0),
              layout_(data_layout::row_major),
              indexing_(sparse_indexing::one_based) {}

    csr_table_impl(const array<byte_t>& data,
                   const array<std::int64_t>& column_indices,
                   const array<std::int64_t>& row_offsets,
                   std::int64_t column_count,
                   data_type dtype,
                   sparse_indexing indexing)
            : meta_(create_metadata(column_count, dtype)),
              data_(data),
              column_indices_(column_indices),
              row_offsets_(row_offsets),
              col_count_(column_count),
              layout_(data_layout::row_major),
              indexing_(indexing) {
        ONEDAL_ASSERT(0 < column_count);
        using error_msg = dal::detail::error_messages;
        std::cout << "in csr_table_impl()" << std::endl << std::flush;

        if (column_count <= 0) {
            throw dal::domain_error(error_msg::cc_leq_zero());
        }

        row_count_ = (row_offsets.get_count() ? row_offsets.get_count() - 1 : 0);
        ONEDAL_ASSERT(0 < row_count_);
        if (row_count_ < 0) {
            throw dal::domain_error(error_msg::rc_leq_zero());
        }

        std::int64_t element_count = get_non_zero_count(row_count_, row_offsets);

        const std::int64_t dtype_size = detail::get_data_type_size(dtype);

        detail::check_mul_overflow(element_count, dtype_size);
        if (data.get_count() != element_count * dtype_size) {
            throw dal::domain_error(error_msg::invalid_data_block_size());
        }

        for (std::int64_t i = 1; i <= row_count_; i++) {
            if (row_offsets[i - 1] > row_offsets[i]) {
                throw dal::domain_error(error_msg::row_offsets_not_ascending());
            }
        }

        if (column_indices.get_count() != element_count) {
            throw dal::domain_error(error_msg::invalid_column_indices_block_size());
        }

        const std::int64_t min_index = (indexing == sparse_indexing::zero_based) ? 0 : 1;
        const std::int64_t max_index =
            (indexing == sparse_indexing::zero_based) ? column_count - 1 : column_count;

        for (std::int64_t i = 0; i < element_count; i++) {
            if (column_indices[i] < min_index) {
                throw dal::domain_error(error_msg::column_indices_lt_min_value());
            }
            if (column_indices[i] > max_index) {
                throw dal::domain_error(error_msg::column_indices_gt_max_value());
            }
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

    std::int64_t get_non_zero_count() const override {
        std::int64_t row_offsets_count = row_offsets_.get_count();
        return (row_offsets_count ? row_offsets_[row_offsets_count - 1] - row_offsets_[0] : 0);
    }

    static std::int64_t get_non_zero_count(const detail::default_host_policy& policy, const std::int64_t row_count, const std::int64_t* row_offsets) {
        if (row_count == 0)
            return 0;

        return (row_offsets[row_count] - row_offsets[0]);
    }

#ifdef ONEDAL_DATA_PARALLEL
    static std::int64_t get_non_zero_count(const detail::data_parallel_policy& policy, const std::int64_t row_count, const std::int64_t* row_offsets) {
        if (row_count == 0)
            return 0;

        auto q = policy.get_queue();
        std::int64_t first_row_offset{0L};
        std::int64_t last_row_offset{0L};
        dal::backend::copy_usm2host(q, &first_row_offset, row_offsets, 1, {}).wait_and_throw();
        dal::backend::copy_usm2host(q, &last_row_offset, row_offsets + row_count, 1, {}).wait_and_throw();

        return (last_row_offset - first_row_offset);
    }
#endif

    static std::int64_t get_non_zero_count(const std::int64_t row_count, const array<std::int64_t>& row_offsets) {
    #ifdef ONEDAL_DATA_PARALLEL
        const auto optional_queue = row_offsets.get_queue();
        if (optional_queue) {
            return csr_table_impl::get_non_zero_count(detail::data_parallel_policy{ optional_queue.value() }, row_count, row_offsets.get_data());
        }
    #endif
        return csr_table_impl::get_non_zero_count(detail::default_host_policy{}, row_count, row_offsets.get_data());
    }

    sparse_indexing get_indexing() const override {
        return indexing_;
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

    array<std::int64_t> get_row_offsets() const override {
        return row_offsets_;
    }

    data_layout get_data_layout() const override {
        return layout_;
    }

    template <typename T>
    void pull_csr_block_template(const detail::default_host_policy& policy,
                                 dal::array<T>& data,
                                 dal::array<std::int64_t>& column_indices,
                                 dal::array<std::int64_t>& row_offsets,
                                 const sparse_indexing& indexing,
                                 const range& rows) const {
        csr_info origin_info{ meta_.get_data_type(0),
                              layout_,
                              row_count_,
                              col_count_,
                              row_offsets_[row_count_] - row_offsets_[0],
                              indexing_ };

        // Overflow is checked here
        check_block_row_range(rows);

        block_info block_info{ rows.start_idx, rows.get_element_count(row_count_), indexing };

        csr_pull_block(policy,
                       origin_info,
                       block_info,
                       data_,
                       column_indices_,
                       row_offsets_,
                       data,
                       column_indices,
                       row_offsets,
                       alloc_kind::host);
    }

    void serialize(detail::output_archive& ar) const override {
        ar(meta_, data_, column_indices_, row_offsets_, col_count_, row_count_, layout_, indexing_);
    }

    void deserialize(detail::input_archive& ar) override {
        ar(meta_, data_, column_indices_, row_offsets_, col_count_, row_count_, layout_, indexing_);
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
    array<std::int64_t> row_offsets_;
    std::int64_t col_count_;
    std::int64_t row_count_;
    data_layout layout_;
    sparse_indexing indexing_;
};

} // namespace oneapi::dal::backend
