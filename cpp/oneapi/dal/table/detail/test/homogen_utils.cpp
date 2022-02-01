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

#include "oneapi/dal/table/detail/homogen_utils.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::detail {

TEST("can get original data from homogen table constructed from const raw pointer") {
    float data[] = { 1.0, 2.0, 3.0, -1.0, -2.0, -3.0 };

    homogen_table t{ data, 2, 3, empty_delete<const float>() };

    auto bytes_array = get_original_data(t);
    REQUIRE(bytes_array.get_data() == (byte_t*)data);
    REQUIRE(bytes_array.get_size() == sizeof(float) * 2 * 3);
    REQUIRE(bytes_array.has_mutable_data() == false);
}

TEST("can get original data from homogen table constructed from builder") {
    auto t = homogen_table_builder{}
                 .set_data_type(data_type::float32)
                 .set_layout(data_layout::row_major)
                 .allocate(3, 2)
                 .build();

    auto bytes_array = get_original_data(t);
    REQUIRE(bytes_array.get_data() == t.get_data());
    REQUIRE(bytes_array.get_size() == sizeof(float) * 2 * 3);
    REQUIRE(bytes_array.has_mutable_data() == true);
}

#ifdef ONEDAL_DATA_PARALLEL
TEST("can get original data from homogen table constructed from array with device data") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    constexpr std::int64_t row_count = 3;
    constexpr std::int64_t col_count = 2;
    auto data = array<float>::zeros(q, row_count * col_count, sycl::usm::alloc::device);

    auto t = homogen_table::wrap(q, data.get_data(), row_count, col_count);

    auto bytes_array = get_original_data(t);
    REQUIRE(bytes_array.get_data() == (const byte_t*)data.get_data());
    REQUIRE(backend::is_device_usm(q, bytes_array.get_data()) == true);
    REQUIRE(bytes_array.get_size() == sizeof(float) * row_count * col_count);
    REQUIRE(bytes_array.has_mutable_data() == false);
}

TEST("can get original data from homogen table constructed from builder with device data") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    constexpr std::int64_t row_count = 3;
    constexpr std::int64_t col_count = 2;
    auto data = array<float>::zeros(q, row_count * col_count, sycl::usm::alloc::device);

    auto t = homogen_table_builder{}
                 .set_layout(data_layout::row_major)
                 .reset(data, row_count, col_count)
                 .build();

    auto bytes_array = get_original_data(t);
    REQUIRE(bytes_array.get_data() == t.get_data());
    REQUIRE(bytes_array.get_data() == (const byte_t*)data.get_data());
    REQUIRE(backend::is_device_usm(q, bytes_array.get_data()) == true);
    REQUIRE(bytes_array.get_size() == sizeof(float) * row_count * col_count);
    REQUIRE(bytes_array.has_mutable_data() == true);
}
#endif

} // namespace oneapi::dal::detail
