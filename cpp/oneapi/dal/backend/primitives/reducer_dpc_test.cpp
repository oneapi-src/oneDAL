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

#include <CL/sycl.hpp>

#include "gtest/gtest.h"

#define ONEAPI_DAL_DATA_PARALLEL

#include "oneapi/dal/data/array.hpp"
#include "oneapi/backend/primitives/reducer.hpp"

TEST(reducer_l2, can_handle_array_of_zeros) {
    cl::sycl::queue q{ cl::sycl::gpu_selector() };

    auto arr = oneapi::dal::array<float>::zeros(q, 35);

    oneapi::dal::l2_reducer_singlepass reducer(q);

    auto res = reducer(arr, 7, 5);

    const float* raw_ptr = res.get_data();

    for(int i = 0; i < 7; i++) ASSERT_EQ(raw_ptr[i], 0.f);
}

