/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

namespace oneapi::dal::backend {

table_metadata create_homogen_metadata(std::int64_t feature_count, data_type dtype);
table_metadata create_homogen_metadata(std::int64_t feature_count,
                                       data_type dtype,
                                       const array<feature_type>& ftypes);

class homogen_table_impl {
public:
    struct host_alloc_t {};

public:
    homogen_table_impl() : row_count_(0), col_count_(0), layout_(data_layout::unknown) {}

    homogen_table_impl(std::int64_t row_count,
                       std::int64_t column_count,
                       const array<byte_t>& data,
                       data_type dtype,
                       data_layout layout)
            : meta_(create_homogen_metadata(column_count, dtype)),
              data_(data),
              row_count_(row_count),
              col_count_(column_count),
              layout_(layout) {
        constructor_checks(dtype);
    }

    homogen_table_impl(std::int64_t row_count,
                       std::int64_t column_count,
                       const array<byte_t>& data,
                       const array<feature_type>& ftypes,
                       data_type dtype,
                       data_layout layout)
            : meta_(create_homogen_metadata(column_count, dtype, ftypes)),
              data_(data),
              row_count_(row_count),
              col_count_(column_count),
              layout_(layout) {
        constructor_checks(dtype);
    }

    std::int64_t get_column_count() const {
        return col_count_;
    }

    std::int64_t get_row_count() const {
        return row_count_;
    }

    const table_metadata& get_metadata() const {
        return meta_;
    }

    const void* get_data() const {
        return data_.get_data();
    }

    data_layout get_data_layout() const {
        return layout_;
    }

    template <typename Data>
    void pull_rows(array<Data>& block, const range& rows) const {
        pull_rows_impl(detail::default_host_policy{}, block, rows, host_alloc_t{});
    }

    template <typename Data>
    void push_rows(const array<Data>& block, const range& rows) {
        push_rows_impl(detail::default_host_policy{}, block, rows);
    }

    template <typename Data>
    void pull_column(array<Data>& block, std::int64_t column_index, const range& rows) const {
        pull_column_impl(detail::default_host_policy{}, block, column_index, rows, host_alloc_t{});
    }

    template <typename Data>
    void push_column(const array<Data>& block, std::int64_t column_index, const range& rows) {
        push_column_impl(detail::default_host_policy{}, block, column_index, rows);
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename Data>
    void pull_rows(sycl::queue& queue,
                   array<Data>& block,
                   const range& rows,
                   const sycl::usm::alloc& kind) const {
        pull_rows_impl(detail::data_parallel_policy{ queue }, block, rows, kind);
    }

    template <typename Data>
    void push_rows(sycl::queue& queue, const array<Data>& block, const range& rows) {
        push_rows_impl(detail::data_parallel_policy{ queue }, block, rows);
    }

    template <typename Data>
    void pull_column(sycl::queue& queue,
                     array<Data>& block,
                     std::int64_t column_index,
                     const range& rows,
                     const sycl::usm::alloc& kind) const {
        pull_column_impl(detail::data_parallel_policy{ queue }, block, column_index, rows, kind);
    }

    template <typename Data>
    void push_column(sycl::queue& queue,
                     const array<Data>& block,
                     std::int64_t column_index,
                     const range& rows) {
        push_column_impl(detail::data_parallel_policy{ queue }, block, column_index, rows);
    }
#endif

private:
    void constructor_checks(data_type dtype) {
        using error_msg = dal::detail::error_messages;

        if (row_count_ <= 0) {
            throw dal::domain_error(error_msg::rc_leq_zero());
        }

        if (col_count_ <= 0) {
            throw dal::domain_error(error_msg::cc_leq_zero());
        }

        detail::check_mul_overflow(row_count_, col_count_);
        const int64_t element_count = row_count_ * col_count_;
        const int64_t dtype_size = detail::get_data_type_size(dtype);

        detail::check_mul_overflow(element_count, dtype_size);
        if (data_.get_count() != element_count * dtype_size) {
            throw dal::domain_error(error_msg::invalid_data_block_size());
        }
        if (layout_ != data_layout::row_major && layout_ != data_layout::column_major) {
            throw dal::domain_error(error_msg::unsupported_data_layout());
        }
    }

    template <typename Policy, typename Data, typename Alloc>
    void pull_rows_impl(const Policy& policy,
                        array<Data>& block,
                        const range& rows,
                        const Alloc& kind) const;

    template <typename Policy, typename Data, typename Alloc>
    void pull_column_impl(const Policy& policy,
                          array<Data>& block,
                          std::int64_t column_index,
                          const range& rows,
                          const Alloc& kind) const;

    template <typename Policy, typename Data>
    void push_rows_impl(const Policy& policy, const array<Data>& block, const range& rows);

    template <typename Policy, typename Data>
    void push_column_impl(const Policy& policy,
                          const array<Data>& block,
                          std::int64_t column_index,
                          const range& rows);

private:
    table_metadata meta_;
    array<byte_t> data_;
    int64_t row_count_;
    int64_t col_count_;
    data_layout layout_;
};

} // namespace oneapi::dal::backend
