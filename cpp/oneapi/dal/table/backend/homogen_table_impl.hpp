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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/detail/common.hpp"

namespace oneapi::dal::backend {

table_metadata create_homogen_metadata(std::int64_t feature_count, data_type dtype);

class homogen_table_impl {
private:
    struct host_alloc_t {};

public:
    homogen_table_impl() : row_count_(0), col_count_(0) {}

    homogen_table_impl(std::int64_t column_count,
                       const array<byte_t>& data,
                       data_type dtype,
                       data_layout layout)
            : meta_(create_homogen_metadata(column_count, dtype)),
              data_(data),
              row_count_(data.get_count() / column_count / detail::get_data_type_size(dtype)),
              col_count_(column_count),
              layout_(layout) {}

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
