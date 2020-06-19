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

#include "gtest/gtest.h"
#include "oneapi/dal/data/table.hpp"
#include "oneapi/dal/data/accessor.hpp"

using namespace oneapi::dal;

TEST(table_test, can_construct_empty_table) {
    table t;

    ASSERT_FALSE(t.has_data());
    ASSERT_EQ(t.get_row_count(), 0);
    ASSERT_EQ(t.get_column_count(), 0);
}

TEST(table_test, can_set_custom_implementation) {
    struct table_impl {
        std::int64_t get_column_count() const noexcept {
            return 10;
        }

        std::int64_t get_row_count() const noexcept {
            return 1000;
        }

        const table_metadata& get_metadata() const noexcept {
            return m;
        }

        void pull_rows(array<float>& a, const range& r) const {}
        void pull_rows(array<double>& a, const range& r) const {}
        void pull_rows(array<std::int32_t>& a, const range& r) const {}
        void push_back_rows(const array<float>&, const range&) {}
        void push_back_rows(const array<double>&, const range&) {}
        void push_back_rows(const array<std::int32_t>&, const range&) {}

        void pull_column(array<float>& a, std::int64_t idx, const range& r) const {}
        void pull_column(array<double>& a, std::int64_t idx, const range& r) const {}
        void pull_column(array<std::int32_t>& a, std::int64_t idx, const range& r) const {}
        void push_back_column(const array<float>&, std::int64_t idx, const range&) {}
        void push_back_column(const array<double>&, std::int64_t idx, const range&) {}
        void push_back_column(const array<std::int32_t>&, std::int64_t idx, const range&) {}

        table_metadata m;
    };

    table t { table_impl{} };
    ASSERT_TRUE(t.has_data());
}

TEST(homogen_table_test, can_construct_empty_table) {
    homogen_table t;

    ASSERT_FALSE(t.has_data());
    ASSERT_EQ(t.get_kind(), homogen_table::kind());
    ASSERT_EQ(t.get_row_count(), 0);
    ASSERT_EQ(t.get_column_count(), 0);
}

TEST(homogen_table_test, can_set_custom_implementation) {
    struct homogen_table_impl {
        std::int64_t get_column_count() const noexcept {
            return 10;
        }

        std::int64_t get_row_count() const noexcept {
            return 1000;
        }

        const homogen_table_metadata& get_metadata() const noexcept {
            return m;
        }

        void pull_rows(array<float>& a, const range& r) const {}
        void pull_rows(array<double>& a, const range& r) const {}
        void pull_rows(array<std::int32_t>& a, const range& r) const {}
        void push_back_rows(const array<float>&, const range&) {}
        void push_back_rows(const array<double>&, const range&) {}
        void push_back_rows(const array<std::int32_t>&, const range&) {}

        void pull_column(array<float>& a, std::int64_t idx, const range& r) const {}
        void pull_column(array<double>& a, std::int64_t idx, const range& r) const {}
        void pull_column(array<std::int32_t>& a, std::int64_t idx, const range& r) const {}
        void push_back_column(const array<float>&, std::int64_t idx, const range&) {}
        void push_back_column(const array<double>&, std::int64_t idx, const range&) {}
        void push_back_column(const array<std::int32_t>&, std::int64_t idx, const range&) {}

        const void* get_data() const { return nullptr; }

        homogen_table_metadata m;
    };

    ASSERT_TRUE(is_homogen_table_impl_v<homogen_table_impl>);

    homogen_table t { homogen_table_impl{} };
    ASSERT_TRUE(t.has_data());
}

TEST(homogen_table_test, can_construct_rowmajor_table_3x2) {
    float data[] = {
        1.f, 2.f,
        3.f, 4.f,
        5.f, 6.f
    };

    homogen_table t { 3, 2, data };

    ASSERT_TRUE(t.has_data());
    ASSERT_EQ(3, t.get_row_count());
    ASSERT_EQ(2, t.get_column_count());

    auto meta = t.get_metadata();
    ASSERT_EQ(homogen_data_layout::row_major, meta.get_data_layout());

    for (std::int64_t i = 0; i < t.get_column_count(); i++) {
        ASSERT_EQ(data_type::float32, meta.get_feature(i).get_data_type());
        ASSERT_EQ(feature_type::ratio, meta.get_feature(i).get_type());
    }

    ASSERT_EQ(data, t.get_data<float>());
    // TODO: now have no ctor to specify feature_type of the data
}

TEST(homogen_table_test, can_construct_colmajor_float64_table) {
    double data[] = {
        1., 2., 3.,
        4., 5., 6.
    };
    homogen_table t { 2, 3, data, homogen_data_layout::column_major };

    ASSERT_TRUE(t.has_data());
    ASSERT_EQ(2, t.get_row_count());
    ASSERT_EQ(3, t.get_column_count());

    auto meta = t.get_metadata();
    ASSERT_EQ(homogen_data_layout::column_major, meta.get_data_layout());

    for (std::int64_t i = 0; i < t.get_column_count(); i++) {
        ASSERT_EQ(data_type::float64, meta.get_feature(i).get_data_type());
        ASSERT_EQ(feature_type::ratio, meta.get_feature(i).get_type());
    }

    ASSERT_EQ(data, t.get_data<double>());
}

TEST(homogen_table_test, can_construct_table_reference) {
    float data[] = {
        1.f, 2.f,
        3.f, 4.f,
        5.f, 6.f
    };

    homogen_table t1 { 3, 2, data };
    homogen_table t2 = t1;

    ASSERT_TRUE(t1.has_data());
    ASSERT_TRUE(t2.has_data());

    ASSERT_EQ(t1.get_row_count(), t2.get_row_count());
    ASSERT_EQ(t1.get_column_count(), t2.get_column_count());
    ASSERT_EQ(t1.get_data<float>(), t2.get_data<float>());
    ASSERT_EQ(data, t1.get_data<float>());
    ASSERT_EQ(data, t2.get_data<float>());

    const auto& m1 = t1.get_metadata();
    const auto& m2 = t2.get_metadata();

    ASSERT_EQ(m1.get_data_layout(), m2.get_data_layout());
    // TODO: replace with metadata objects comparison
    for (std::int64_t i = 0; i < t1.get_column_count(); i++) {
        ASSERT_EQ(m1.get_feature(i).get_data_type(), m2.get_feature(i).get_data_type());
        ASSERT_EQ(m1.get_feature(i).get_type(), m2.get_feature(i).get_type());
    }
}

TEST(homogen_table_test, can_construct_table_with_move) {
    float data[] = {
        1.f, 2.f,
        3.f, 4.f,
        5.f, 6.f
    };

    homogen_table t1 { 3, 2, data };
    homogen_table t2 = std::move(t1);

    ASSERT_FALSE(t1.has_data());
    ASSERT_TRUE(t2.has_data());

    ASSERT_EQ(3, t2.get_row_count());
    ASSERT_EQ(2, t2.get_column_count());

    const auto& m1 = t1.get_metadata();
    const auto& m2 = t2.get_metadata();

    ASSERT_EQ(homogen_data_layout::row_major, m2.get_data_layout());
    ASSERT_EQ(data_type::float32, m2.get_feature(0).get_data_type());
    ASSERT_EQ(data_type::float32, m2.get_feature(1).get_data_type());
    ASSERT_EQ(data, t2.get_data<float>());
}

TEST(homogen_table_test, can_assign_two_table_references) {
    float data_float[] = {
        1.f, 2.f,
        3.f, 4.f,
        5.f, 6.f
    };

    std::int32_t data_int[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    };

    homogen_table t1 { 3, 2, data_float };
    homogen_table t2 { 4, 3, data_int };

    t1 = t2;

    ASSERT_TRUE(t1.has_data());
    ASSERT_TRUE(t2.has_data());

    ASSERT_EQ(4, t1.get_row_count());
    ASSERT_EQ(3, t1.get_column_count());
    ASSERT_EQ(data_type::int32, t1.get_metadata().get_feature(0).get_data_type());
    ASSERT_EQ(data_int, t1.get_data<std::int32_t>());

    ASSERT_EQ(4, t2.get_row_count());
    ASSERT_EQ(3, t2.get_column_count());
    ASSERT_EQ(data_type::int32, t2.get_metadata().get_feature(0).get_data_type());
    ASSERT_EQ(data_int, t2.get_data<std::int32_t>());
}

TEST(homogen_table_test, can_move_assigned_table_reference) {
    float data_float[] = {
        1.f, 2.f,
        3.f, 4.f,
        5.f, 6.f
    };

    std::int32_t data_int[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    };

    homogen_table t1 { 3, 2, data_float };
    homogen_table t2 { 4, 3, data_int };

    t1 = std::move(t2);

    ASSERT_TRUE(t1.has_data());
    ASSERT_EQ(4, t1.get_row_count());
    ASSERT_EQ(3, t1.get_column_count());
    ASSERT_EQ(data_type::int32, t1.get_metadata().get_feature(0).get_data_type());
    ASSERT_EQ(data_int, t1.get_data<std::int32_t>());
}

TEST(homogen_table_test, can_upcast_table) {
    float data_float[] = {
        1.f, 2.f,
        3.f, 4.f,
        5.f, 6.f
    };

    table t = homogen_table { 3, 2, data_float };

    ASSERT_TRUE(t.has_data());
    ASSERT_EQ(3, t.get_row_count());
    ASSERT_EQ(2, t.get_column_count());
    ASSERT_EQ(data_type::float32, t.get_metadata().get_feature(0).get_data_type());
}

TEST(homogen_table_test, can_read_table_data_via_row_accessor) {
    double data[] = {
        1.0, 2.0, 3.0,
        -1.0, -2.0, -3.0
    };

    homogen_table t { 2, 3, data };
    const auto rows_block = row_accessor<const double>(t).pull({0, -1});

    ASSERT_EQ(t.get_row_count() * t.get_column_count(), rows_block.get_size());
    ASSERT_EQ(data, rows_block.get_data());
    ASSERT_FALSE(rows_block.is_data_owner());

    for (std::int64_t i = 0; i < rows_block.get_size(); i++) {
        ASSERT_EQ(rows_block[i], data[i]);
    }
}

TEST(homogen_table_test, can_read_table_data_via_row_accessor_with_conversion) {
    float data[] = {
         1.0f,  2.0f,  3.0f,
        -1.0f, -2.0f, -3.0f
    };

    homogen_table t { 2, 3, data };
    auto rows_block = row_accessor<const double>(t).pull({0, -1});

    ASSERT_EQ(t.get_row_count() * t.get_column_count(), rows_block.get_size());
    ASSERT_NE((void*)data, (void*)rows_block.get_data());
    ASSERT_TRUE(rows_block.is_data_owner());

    for (std::int64_t i = 0; i < rows_block.get_size(); i++) {
        ASSERT_DOUBLE_EQ(rows_block[i], static_cast<double>(data[i]));
    }
}

TEST(homogen_table_test, can_read_table_data_via_row_accessor_and_array_outside) {
    float data[] = {
         1.0f,  2.0f,  3.0f,
        -1.0f, -2.0f, -3.0f
    };

    homogen_table t { 2, 3, data };
    array<float> arr { 10 };

    auto rows_ptr = row_accessor<const float>(t).pull(arr, {0, -1});

    ASSERT_EQ(t.get_row_count() * t.get_column_count(), arr.get_size());
    ASSERT_EQ(arr.get_capacity(), 10);
    ASSERT_FALSE(arr.is_data_owner());

    ASSERT_EQ(data, rows_ptr);
    ASSERT_EQ(data, arr.get_data());

    auto data_ptr = arr.get_data();
    for (std::int64_t i = 0; i < arr.get_size(); i++) {
        ASSERT_EQ(rows_ptr[i], data[i]);
        ASSERT_EQ(data_ptr[i], data[i]);
    }
}

