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

#include "oneapi/dal/table/common.hpp"
#include "gtest/gtest.h"

using namespace oneapi::dal;

TEST(table_test, can_construct_empty_table) {
    table t;

    ASSERT_FALSE(t.has_data());
    ASSERT_EQ(t.get_row_count(), 0);
    ASSERT_EQ(t.get_column_count(), 0);
}

TEST(table_test, can_set_custom_implementation) {
    struct table_impl {
        const std::int64_t kind = 123456;

        std::int64_t get_column_count() const noexcept {
            return 10;
        }

        std::int64_t get_row_count() const noexcept {
            return 1000;
        }

        const table_metadata& get_metadata() const noexcept {
            return m;
        }

        std::int64_t get_kind() const {
            return kind;
        }

        void pull_rows(array<float>& a, const range& r) const {}
        void pull_rows(array<double>& a, const range& r) const {}
        void pull_rows(array<std::int32_t>& a, const range& r) const {}
        void push_rows(const array<float>&, const range&) {}
        void push_rows(const array<double>&, const range&) {}
        void push_rows(const array<std::int32_t>&, const range&) {}

        void pull_column(array<float>& a, std::int64_t idx, const range& r) const {}
        void pull_column(array<double>& a, std::int64_t idx, const range& r) const {}
        void pull_column(array<std::int32_t>& a, std::int64_t idx, const range& r) const {}
        void push_column(const array<float>&, std::int64_t idx, const range&) {}
        void push_column(const array<double>&, std::int64_t idx, const range&) {}
        void push_column(const array<std::int32_t>&, std::int64_t idx, const range&) {}

        table_metadata m;
    };

    table t{ table_impl{} };
    ASSERT_TRUE(t.has_data());
    ASSERT_EQ(t.get_kind(), table_impl{}.get_kind());
}
