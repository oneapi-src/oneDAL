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
#include "oneapi/dal/table/backend/heterogen_kernels.hpp"
#include "oneapi/dal/table/detail/csr_access_iface.hpp"
#include "oneapi/dal/backend/serialization.hpp"

namespace oneapi::dal::backend {

/*class heterogen_table_impl : public detail::heterogen_table_template<heterogen_table_impl>,
                             public ONEDAL_SERIALIZABLE(heterogen_table_id) {
public:
    heterogen_table_impl()
            : col_count_(0),
             {}

    heterogen_table_impl(std::int64_t row_count,
                   std::int64_t column_count,
                   data_type dtype)
            : meta_(create_metadata(column_count, dtype)),
              data_(data),
              column_indices_(column_indices),
              row_offsets_(row_offsets),
              col_count_(column_count)  {
        using error_msg = dal::detail::error_messages;

        const std::int64_t element_count = column_indices.get_count();
        const std::int64_t dtype_size = detail::get_data_type_size(dtype);

        detail::check_mul_overflow(element_count, dtype_size);
        if (data.get_count() != element_count * dtype_size) {
            throw dal::domain_error(error_msg::invalid_data_block_size());
        }

        const std::int64_t* const row_offsets_ptr = row_offsets.get_data();
        if (!std::is_sorted(row_offsets_ptr, row_offsets_ptr + row_offsets.get_count())) {
            throw dal::domain_error(error_msg::row_offsets_not_ascending());
        }

        const std::int64_t min_index = (indexing == sparse_indexing::zero_based) ? 0 : 1;
        const std::int64_t max_index =
            (indexing == sparse_indexing::zero_based) ? column_count - 1 : column_count;

        const auto status = backend::check_bounds(column_indices, min_index, max_index);
        if (status == backend::out_of_bound_type::less_than_min) {
            throw dal::domain_error(error_msg::column_indices_lt_min_value());
        }
        if (status == backend::out_of_bound_type::greater_than_max) {
            throw dal::domain_error(error_msg::column_indices_gt_max_value());
        }
    }

#ifdef ONEDAL_DATA_PARALLEL
    csr_table_impl(const detail::data_parallel_policy& policy,
                   const array<byte_t>& data,
                   const array<std::int64_t>& column_indices,
                   const array<std::int64_t>& row_offsets,
                   std::int64_t column_count,
                   data_type dtype,
                   sparse_indexing indexing,
                   const std::vector<sycl::event>& dependencies)
            : meta_(create_metadata(column_count, dtype)),
              data_(data),
              column_indices_(column_indices),
              row_offsets_(row_offsets),
              col_count_(column_count),
              row_count_(row_offsets.get_count() - 1),
              layout_(data_layout::row_major),
              indexing_(indexing) {
        using error_msg = dal::detail::error_messages;

        const std::int64_t element_count = column_indices.get_count();
        const std::int64_t dtype_size = detail::get_data_type_size(dtype);

        detail::check_mul_overflow(element_count, dtype_size);
        if (data.get_count() != element_count * dtype_size) {
            throw dal::domain_error(error_msg::invalid_data_block_size());
        }

        if (!backend::is_sorted(row_offsets, dependencies)) {
            throw dal::domain_error(error_msg::row_offsets_not_ascending());
        }

        const std::int64_t min_index = (indexing == sparse_indexing::zero_based) ? 0 : 1;
        const std::int64_t max_index =
            (indexing == sparse_indexing::zero_based) ? column_count - 1 : column_count;

        const auto status =
            backend::check_bounds(column_indices, min_index, max_index, dependencies);
        if (status == backend::out_of_bound_type::less_than_min) {
            throw dal::domain_error(error_msg::column_indices_lt_min_value());
        }
        if (status == backend::out_of_bound_type::greater_than_max) {
            throw dal::domain_error(error_msg::column_indices_gt_max_value());
        }
    }
#endif

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
    std::int64_t col_count_;
    std::int64_t row_count_;
};*/

} // namespace oneapi::dal::backend
