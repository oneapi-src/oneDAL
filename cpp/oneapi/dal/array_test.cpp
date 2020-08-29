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

#include "oneapi/dal/array.hpp"
#include "gtest/gtest.h"

using namespace oneapi::dal;
using std::int32_t;

TEST(array_test, can_construct_empty_array) {
    array<float> arr;

    ASSERT_EQ(arr.get_count(), 0);
    ASSERT_FALSE(arr.has_mutable_data());
}

TEST(array_test, can_construct_array_of_zeros) {
    auto arr = array<float>::zeros(5);

    ASSERT_EQ(arr.get_count(), 5);
    ASSERT_TRUE(arr.has_mutable_data());

    for (int32_t i = 0; i < arr.get_count(); i++) {
        ASSERT_FLOAT_EQ(arr[i], 0.0f);
    }
}

TEST(array_test, can_construct_array_of_ones) {
    auto arr = array<float>::full(5, 1.0f);

    ASSERT_EQ(arr.get_count(), 5);
    ASSERT_TRUE(arr.has_mutable_data());

    for (int32_t i = 0; i < arr.get_count(); i++) {
        ASSERT_FLOAT_EQ(arr[i], 1.0f);
    }
}

TEST(array_test, can_construct_array_from_raw_pointer) {
    constexpr int64_t size = 10;
    auto ptr = new float[size];
    for (int64_t i = 0; i < size; i++) {
        ptr[i] = float(i);
    }

    array arr(ptr, size, [](auto ptr) {
        delete[] ptr;
    });

    ASSERT_EQ(arr.get_count(), size);
    ASSERT_TRUE(arr.has_mutable_data());

    for (int32_t i = 0; i < arr.get_count(); i++) {
        ASSERT_FLOAT_EQ(arr[i], ptr[i]);
    }
}

TEST(array_test, can_construct_array_reference) {
    auto arr = array<float>::zeros(5);
    array<float> arr2 = arr;

    ASSERT_EQ(arr.get_count(), arr2.get_count());
    ASSERT_EQ(arr.get_data(), arr2.get_data());
    ASSERT_EQ(arr.has_mutable_data(), arr2.has_mutable_data());
    ASSERT_EQ(arr.get_mutable_data(), arr2.get_mutable_data());

    for (int32_t i = 0; i < arr.get_count(); i++) {
        ASSERT_FLOAT_EQ(arr[i], arr2[i]);
    }
}

TEST(array_test, can_reset_array) {
    auto arr = array<float>::zeros(5);
    arr.reset();

    ASSERT_EQ(arr.get_count(), 0);
    ASSERT_FALSE(arr.has_mutable_data());
}

TEST(array_test, can_reset_array_with_bigger_size) {
    auto arr = array<float>::zeros(5);
    arr.reset(10);

    ASSERT_EQ(arr.get_count(), 10);
    ASSERT_TRUE(arr.has_mutable_data());
}

TEST(array_test, can_reset_array_with_smaller_size) {
    auto arr = array<float>::zeros(5);
    arr.reset(4);

    ASSERT_EQ(arr.get_count(), 4);
    ASSERT_TRUE(arr.has_mutable_data());
}

TEST(array_test, can_reset_array_with_raw_pointer) {
    auto arr = array<float>::zeros(5);

    constexpr int64_t size = 10;
    auto ptr = new float[size];
    arr.reset(ptr, size, [](auto ptr) {
        delete[] ptr;
    });

    ASSERT_EQ(arr.get_count(), size);
    ASSERT_TRUE(arr.has_mutable_data());
    ASSERT_EQ(arr.get_mutable_data(), ptr);
    ASSERT_EQ(arr.get_data(), ptr);
}

TEST(array_test, can_reset_array_with_non_owning_raw_pointer) {
    auto arr = array<float>::zeros(5);

    constexpr int64_t size = 10;
    const float* ptr = new float[size];
    arr.reset(array<float>(), ptr, size);

    ASSERT_EQ(arr.get_count(), size);
    ASSERT_EQ(arr.get_data(), ptr);
    ASSERT_FALSE(arr.has_mutable_data());
    // ASSERT_THROW(arr.get_mutable_data(), std::bad_variant_access);

    delete[] ptr;
}

TEST(array_test, can_make_owning_array_from_non_owning_readonly) {
    array<float> arr;

    float data[] = { 1.f, 2.f, 3.f };
    arr.reset(array<float>(), (const float*)(data), 3);

    ASSERT_EQ(arr.get_count(), 3);
    ASSERT_EQ(arr.get_data(), data);
    ASSERT_FALSE(arr.has_mutable_data());
    // ASSERT_THROW(arr.get_mutable_data(), std::bad_variant_access);

    arr.need_mutable_data();

    ASSERT_EQ(arr.get_count(), 3);
    ASSERT_NE(arr.get_data(), data);
    ASSERT_NE(arr.get_mutable_data(), data);
    ASSERT_TRUE(arr.has_mutable_data());

    for (int64_t i = 0; i < arr.get_count(); i++) {
        ASSERT_FLOAT_EQ(arr[i], data[i]);
    }
}

TEST(array_test, can_construct_non_owning_read_write_array) {
    float data[] = { 1.0f, 2.0f, 3.0f };
    array<float> arr{ data, 3, [](auto) {} };

    ASSERT_EQ(arr.get_count(), 3);
    ASSERT_EQ(arr.get_data(), data);
    ASSERT_TRUE(arr.has_mutable_data());
}

TEST(array_test, can_construct_non_owning_read_only_array) {
    float data[] = { 1.0f, 2.0f, 3.0f };
    array<float> arr{ array<float>(), static_cast<const float*>(data), 3 };

    ASSERT_EQ(arr.get_count(), 3);
    ASSERT_EQ(arr.get_data(), data);
    ASSERT_FALSE(arr.has_mutable_data());
}

TEST(array_test, can_wrap_const_data) {
    const float data[] = { 1.0f, 2.0f, 3.0f };
    auto arr = array<float>::wrap(data, 3);

    ASSERT_EQ(arr.get_count(), 3);
    ASSERT_EQ(arr.get_data(), data);
    ASSERT_FALSE(arr.has_mutable_data());
}

TEST(array_test, can_wrap_const_data_with_offset_and_deleter) {
    float* data = new float[3];
    data[0] = 0.0f;
    data[1] = 1.0f;
    data[2] = 2.0f;

    const float* cdata = data;
    auto arr = array<float>(cdata, 2, [](const float* ptr) {
        delete[] ptr;
    });

    ASSERT_EQ(arr.get_count(), 2);
    ASSERT_EQ(arr.get_data(), cdata);
    ASSERT_FALSE(arr.has_mutable_data());
}
