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

namespace oneapi::dal::test::engine {

class dummy_table_impl : public detail::generic_table_template<dummy_table_impl> {
public:
    explicit dummy_table_impl(std::int64_t row_count, std::int64_t column_count)
            : row_count_(row_count),
              column_count_(column_count) {}

    std::int64_t get_kind() const override {
        return -1;
    }

    std::int64_t get_column_count() const override {
        return column_count_;
    }

    std::int64_t get_row_count() const override {
        return row_count_;
    }

    const table_metadata& get_metadata() const override {
        return metadata_;
    }

    data_layout get_data_layout() const override {
        return data_layout::column_major;
    }

    template <typename Data>
    void pull_rows_template(const detail::default_host_policy&,
                            array<Data>& block,
                            const range& row_range) const {
        block.reset();
    }

    template <typename Data>
    void pull_column_template(const detail::default_host_policy&,
                              array<Data>& block,
                              std::int64_t column_index,
                              const range& row_range) const {
        block.reset();
    }

    template <typename Data>
    void pull_csr_block_template(const detail::default_host_policy& policy,
                                 dal::array<Data>& data,
                                 dal::array<std::int64_t>& column_indices,
                                 dal::array<std::int64_t>& row_offsets,
                                 const sparse_indexing& indexing,
                                 const range& row_range) const {
        data.reset();
        column_indices.reset();
        row_offsets.reset();
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename Data>
    void pull_rows_template(const detail::data_parallel_policy&,
                            array<Data>& block,
                            const range& row_range,
                            const sycl::usm::alloc&) const {
        block.reset();
    }

    template <typename Data>
    void pull_column_template(const detail::data_parallel_policy&,
                              array<Data>& block,
                              std::int64_t column_index,
                              const range& row_range,
                              const sycl::usm::alloc&) const {
        block.reset();
    }

    template <typename Data>
    void pull_csr_block_template(const detail::data_parallel_policy&,
                                 dal::array<Data>& data,
                                 dal::array<std::int64_t>& column_indices,
                                 dal::array<std::int64_t>& row_offsets,
                                 const sparse_indexing& indexing,
                                 const range& row_range,
                                 sycl::usm::alloc alloc) const {
        data.reset();
        column_indices.reset();
        row_offsets.reset();
    }
#endif

private:
    table_metadata metadata_;
    std::int64_t row_count_;
    std::int64_t column_count_;
};

class dummy_table : public table {
public:
    dummy_table(std::int64_t row_count, std::int64_t column_count)
            : table(new dummy_table_impl{ row_count, column_count }) {}
};

} // namespace oneapi::dal::test::engine
