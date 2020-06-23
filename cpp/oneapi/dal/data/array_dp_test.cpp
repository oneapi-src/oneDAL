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
#include <cl/sycl.hpp>

#define ENABLE_DATA_PARALLEL_EXECUTION
#include "oneapi/dal/data/array.hpp"

using namespace oneapi::dal;
using std::int32_t;

TEST(array_dp_test, can_construct_array_of_zeros) {
    cl::sycl::queue q { cl::sycl::gpu_selector() };

    auto arr = array<float>::zeros(q, 5);

    ASSERT_EQ(arr.get_size(), 5);
    ASSERT_EQ(arr.get_capacity(), 5);
    ASSERT_TRUE(arr.is_data_owner());
    ASSERT_TRUE(arr.has_mutable_data());

    for (int32_t i = 0; i < arr.get_size(); i++) {
        ASSERT_FLOAT_EQ(arr[i], 0.0f);
    }
}
