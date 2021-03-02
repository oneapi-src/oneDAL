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

#include "gtest/gtest.h"

#define ONEDAL_DATA_PARALLEL
#include "oneapi/dal/array.hpp"

using namespace oneapi::dal;
using std::int32_t;

TEST(array_dpc_test, can_construct_array_of_zeros) {
    sycl::queue q{ sycl::gpu_selector() };

    auto arr = array<float>::zeros(q, 5);

    ASSERT_EQ(arr.get_count(), 5);
    ASSERT_TRUE(arr.has_mutable_data());

    for (int32_t i = 0; i < arr.get_count(); i++) {
        ASSERT_FLOAT_EQ(arr[i], 0.0f);
    }
}

TEST(array_dpc_test, can_construct_array_of_ones) {
    sycl::queue q{ sycl::gpu_selector() };

    auto arr = array<float>::full(5, 1.0f);

    ASSERT_EQ(arr.get_count(), 5);
    ASSERT_TRUE(arr.has_mutable_data());

    for (int32_t i = 0; i < arr.get_count(); i++) {
        ASSERT_FLOAT_EQ(arr[i], 1.0f);
    }
}

TEST(array_dpc_test, can_construct_device_array_without_initialization) {
    sycl::queue q{ sycl::gpu_selector() };

    auto arr = array<float>::empty(q, 10, sycl::usm::alloc::device);

    ASSERT_EQ(arr.get_count(), 10);
    ASSERT_TRUE(arr.has_mutable_data());

    ASSERT_EQ(sycl::get_pointer_type(arr.get_data(), q.get_context()), sycl::usm::alloc::device);
}

TEST(array_dpc_test, can_construct_array_with_events) {
    sycl::queue q{ sycl::gpu_selector() };

    constexpr std::int64_t count = 10;

    auto* data = sycl::malloc_shared<float>(count, q);
    q.submit([&](sycl::handler& cgh) {
         cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
             data[idx[0]] = idx[0];
         });
     }).wait();

    array<float> arr{ q, data, 10, make_default_delete<float>(q) };

    ASSERT_EQ(arr.get_count(), 10);
    ASSERT_TRUE(arr.has_mutable_data());

    for (int32_t i = 0; i < arr.get_count(); i++) {
        ASSERT_FLOAT_EQ(arr[i], float(i));
    }
}

TEST(array_dpc_test, can_reset_array_with_bigger_size) {
    sycl::queue q{ sycl::gpu_selector() };

    auto arr = array<float>::zeros(q, 5);
    arr.reset(q, 10);

    ASSERT_EQ(arr.get_count(), 10);
    ASSERT_TRUE(arr.has_mutable_data());
}

TEST(array_dpc_test, can_reset_array_with_smaller_size) {
    sycl::queue q{ sycl::gpu_selector() };

    auto arr = array<float>::zeros(q, 5);
    arr.reset(q, 4);

    ASSERT_EQ(arr.get_count(), 4);
    ASSERT_TRUE(arr.has_mutable_data());
}

TEST(array_dpc_test, can_reset_array_with_raw_pointer) {
    sycl::queue q{ sycl::gpu_selector() };

    auto arr = array<float>::zeros(q, 5);

    constexpr int64_t count = 10;
    auto* data = sycl::malloc_shared<float>(count, q);
    q.submit([&](sycl::handler& cgh) {
         cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
             data[idx[0]] = idx[0];
         });
     }).wait();

    arr.reset(data, count, make_default_delete<float>(q));

    ASSERT_EQ(arr.get_size(), count * sizeof(float));
    ASSERT_EQ(arr.get_count(), count);
    ASSERT_TRUE(arr.has_mutable_data());
    ASSERT_EQ(arr.get_mutable_data(), data);
    ASSERT_EQ(arr.get_data(), data);
}

TEST(array_dpc_test, can_wrap_const_data_with_offset_and_deleter) {
    sycl::queue q{ sycl::gpu_selector() };
    constexpr int64_t count = 3;

    auto data = sycl::malloc_shared<float>(count, q);
    q.submit([&](sycl::handler& cgh) {
         cgh.parallel_for(sycl::range<1>(count), [=](sycl::id<1> idx) {
             data[idx[0]] = idx;
         });
     }).wait();

    const float* cdata = data;
    auto arr = array<float>(q, cdata, 2, make_default_delete<const float>(q));

    ASSERT_EQ(arr.get_count(), 2);
    ASSERT_EQ(arr.get_data(), cdata);
    ASSERT_FALSE(arr.has_mutable_data());
}
