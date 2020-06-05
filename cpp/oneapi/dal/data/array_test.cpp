/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
#include "oneapi/dal/data/array.hpp"

using namespace dal;
using std::int32_t;

TEST(array_test, can_construct_empty_array) {
    array<float> arr;

    ASSERT_EQ(arr.get_size(), 0);
    ASSERT_EQ(arr.get_capacity(), 0);
    ASSERT_FALSE(arr.is_data_owner());
    ASSERT_FALSE(arr.has_mutable_data());
}

TEST(array_test, can_construct_array_of_zeros) {
    array<float> arr(5);

    ASSERT_EQ(arr.get_size(), 5);
    ASSERT_EQ(arr.get_capacity(), 5);
    ASSERT_TRUE(arr.is_data_owner());
    ASSERT_TRUE(arr.has_mutable_data());

    for (int32_t i = 0; i < arr.get_size(); i++) {
        ASSERT_FLOAT_EQ(arr[i], 0.0f);
    }
}

TEST(array_test, can_construct_array_of_ones) {
    array arr(5, 1.0f);

    ASSERT_EQ(arr.get_size(), 5);
    ASSERT_EQ(arr.get_capacity(), 5);
    ASSERT_TRUE(arr.is_data_owner());
    ASSERT_TRUE(arr.has_mutable_data());

    for (int32_t i = 0; i < arr.get_size(); i++) {
        ASSERT_FLOAT_EQ(arr[i], 1.0f);
    }
}

TEST(array_test, can_construct_array_from_raw_pointer) {
    constexpr int64_t size = 10;
    auto ptr = new float[size];
    for (int64_t i = 0; i < size; i++) {
        ptr[i] = float(i);
    }

    array arr(ptr, size, [](auto ptr){ delete[] ptr; });

    ASSERT_EQ(arr.get_size(), size);
    ASSERT_EQ(arr.get_capacity(), size);
    ASSERT_TRUE(arr.is_data_owner());
    ASSERT_TRUE(arr.has_mutable_data());

    for (int32_t i = 0; i < arr.get_size(); i++) {
        ASSERT_FLOAT_EQ(arr[i], ptr[i]);
    }
}

TEST(array_test, can_construct_array_reference) {
    array<float> arr(5);
    array<float> arr2 = arr;

    ASSERT_EQ(arr.get_size(), arr2.get_size());
    ASSERT_EQ(arr.get_capacity(), arr2.get_capacity());
    ASSERT_EQ(arr.is_data_owner(), arr2.is_data_owner());
    ASSERT_EQ(arr.get_data(), arr2.get_data());
    ASSERT_EQ(arr.has_mutable_data(), arr2.has_mutable_data());
    ASSERT_EQ(arr.get_mutable_data(), arr2.get_mutable_data());

    for (int32_t i = 0; i < arr.get_size(); i++) {
        ASSERT_FLOAT_EQ(arr[i], arr2[i]);
    }
}

TEST(array_test, can_reset_array) {
    array<float> arr(5);
    arr.reset();

    ASSERT_EQ(arr.get_size(), 0);
    ASSERT_EQ(arr.get_capacity(), 0);
    ASSERT_FALSE(arr.is_data_owner());
    ASSERT_FALSE(arr.has_mutable_data());
}

TEST(array_test, can_reset_array_with_bigger_size) {
    array<float> arr(5);
    arr.reset(10);

    ASSERT_EQ(arr.get_size(), 10);
    ASSERT_EQ(arr.get_capacity(), 10);
    ASSERT_TRUE(arr.is_data_owner());
    ASSERT_TRUE(arr.has_mutable_data());
}

TEST(array_test, can_reset_array_with_smaller_size) {
    array<float> arr(5);
    arr.reset(4);

    ASSERT_EQ(arr.get_size(), 4);
    ASSERT_EQ(arr.get_capacity(), 4);
    ASSERT_TRUE(arr.is_data_owner());
    ASSERT_TRUE(arr.has_mutable_data());
}

TEST(array_test, can_reset_array_with_raw_pointer) {
    array<float> arr(5);

    constexpr int64_t size = 10;
    auto ptr = new float[size];
    arr.reset(ptr, size, [](auto ptr){ delete[] ptr; });

    ASSERT_EQ(arr.get_size(), size);
    ASSERT_EQ(arr.get_capacity(), size);
    ASSERT_TRUE(arr.is_data_owner());
    ASSERT_TRUE(arr.has_mutable_data());
    ASSERT_EQ(arr.get_mutable_data(), ptr);
    ASSERT_EQ(arr.get_data(), ptr);
}

TEST(array_test, can_reset_array_with_non_owning_raw_pointer) {
    array<float> arr(5);

    constexpr int64_t size = 10;
    const float* ptr = new float[size];
    arr.reset_not_owning(ptr, size);

    ASSERT_EQ(arr.get_size(), size);
    ASSERT_EQ(arr.get_capacity(), 5);
    ASSERT_FALSE(arr.is_data_owner());
    ASSERT_EQ(arr.get_data(), ptr);
    ASSERT_FALSE(arr.has_mutable_data());
    ASSERT_THROW(arr.get_mutable_data(), std::bad_variant_access);

    delete[] ptr;
}

TEST(array_test, can_resize_array_with_bigger_size) {
    array<float> arr(5);
    arr.resize(10);

    ASSERT_EQ(arr.get_size(), 10);
    ASSERT_EQ(arr.get_capacity(), 10);
    ASSERT_TRUE(arr.is_data_owner());
    ASSERT_TRUE(arr.has_mutable_data());
}

TEST(array_test, can_resize_array_with_smaller_size) {
    array<float> arr(5);
    arr.resize(4);

    ASSERT_EQ(arr.get_size(), 4);
    ASSERT_EQ(arr.get_capacity(), 5);
    ASSERT_TRUE(arr.is_data_owner());
    ASSERT_TRUE(arr.has_mutable_data());
}

TEST(array_test, can_make_owning_array_from_non_owning) {
    array<float> arr;

    float data[] = { 1.f, 2.f, 3.f };
    arr.reset_not_owning(data, 3);

    ASSERT_EQ(arr.get_size(), 3);
    ASSERT_EQ(arr.get_capacity(), 0);
    ASSERT_EQ(arr.get_data(), data);
    ASSERT_EQ(arr.get_mutable_data(), data);
    ASSERT_TRUE(arr.has_mutable_data());
    ASSERT_FALSE(arr.is_data_owner());

    arr.unique();

    ASSERT_EQ(arr.get_size(), 3);
    ASSERT_EQ(arr.get_capacity(), 3);
    ASSERT_NE(arr.get_data(), data);
    ASSERT_NE(arr.get_mutable_data(), data);
    ASSERT_TRUE(arr.has_mutable_data());
    ASSERT_TRUE(arr.is_data_owner());

    for (int64_t i = 0; i < arr.get_size(); i++) {
        ASSERT_FLOAT_EQ(arr[i], data[i]);
    }
}

TEST(array_test, can_make_owning_array_from_non_owning_readonly) {
    array<float> arr;

    float data[] = { 1.f, 2.f, 3.f };
    arr.reset_not_owning<const float*>(data, 3);

    ASSERT_EQ(arr.get_size(), 3);
    ASSERT_EQ(arr.get_capacity(), 0);
    ASSERT_EQ(arr.get_data(), data);
    ASSERT_FALSE(arr.has_mutable_data());
    ASSERT_THROW(arr.get_mutable_data(), std::bad_variant_access);
    ASSERT_FALSE(arr.is_data_owner());

    arr.unique();

    ASSERT_EQ(arr.get_size(), 3);
    ASSERT_EQ(arr.get_capacity(), 3);
    ASSERT_NE(arr.get_data(), data);
    ASSERT_NE(arr.get_mutable_data(), data);
    ASSERT_TRUE(arr.has_mutable_data());
    ASSERT_TRUE(arr.is_data_owner());

    for (int64_t i = 0; i < arr.get_size(); i++) {
        ASSERT_FLOAT_EQ(arr[i], data[i]);
    }
}
