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

class dummy_homogen_table_impl {
public:
    explicit dummy_homogen_table_impl(std::int64_t row_count, std::int64_t column_count)
            : row_count_(row_count),
              column_count_(column_count) {}

    std::int64_t get_column_count() const noexcept {
        return column_count_;
    }

    std::int64_t get_row_count() const noexcept {
        return row_count_;
    }

    template <typename Data>
    void pull_rows(array<Data>& block, const range&) const {
        block.reset();
    }

    template <typename Data>
    void pull_column(array<Data>& block, std::int64_t, const range&) const {
        block.reset();
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename Data>
    void pull_rows(sycl::queue&, array<Data>& block, const range&, const sycl::usm::alloc&) const {
        block.reset();
    }

    template <typename Data>
    void pull_column(sycl::queue&,
                     array<Data>& block,
                     std::int64_t,
                     const range&,
                     const sycl::usm::alloc&) const {
        block.reset();
    }
#endif

    const void* get_data() const {
        return nullptr;
    }

    const table_metadata& get_metadata() const {
        return metadata_;
    }

    data_layout get_data_layout() const {
        return data_layout::column_major;
    }

private:
    table_metadata metadata_;
    std::int64_t row_count_;
    std::int64_t column_count_;
};

class dummy_homogen_table : public homogen_table {
public:
    dummy_homogen_table(std::int64_t row_count, std::int64_t column_count)
            : homogen_table(dummy_homogen_table_impl{ row_count, column_count }) {}
};

} // namespace oneapi::dal::test::engine
