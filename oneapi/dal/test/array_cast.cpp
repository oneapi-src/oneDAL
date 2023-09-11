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
#include "oneapi/dal/detail/array_utils.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/linalg.hpp"

namespace oneapi::dal::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

TEST("can reinterpret_cast empty array") {
    array<float> arr;

    const auto arr_cast = detail::reinterpret_array_cast<std::int32_t>(arr);

    REQUIRE(arr_cast.get_data() == nullptr);
    REQUIRE(arr_cast.get_count() == 0);
}

TEST("can reinterpret_cast immutable array") {
    constexpr std::int64_t count = 2;
    const float data[count] = { -1.0, 2.0 };
    const auto arr = array<float>::wrap(data, count);

    const auto arr_cast = detail::reinterpret_array_cast<byte_t>(arr);

    REQUIRE(reinterpret_cast<const void*>(arr_cast.get_data()) ==
            reinterpret_cast<const void*>(data));
    REQUIRE(arr_cast.has_mutable_data() == false);
    REQUIRE(arr_cast.get_count() == count * sizeof(float));
    REQUIRE(arr_cast.get_size() == arr.get_size());
}

TEST("can reinterpret_cast mutable array") {
    constexpr std::int64_t count = 2;
    float data[count] = { -1.0, 2.0 };
    const auto arr = array<float>::wrap(data, count);

    const auto arr_cast = detail::reinterpret_array_cast<byte_t>(arr);

    REQUIRE(reinterpret_cast<const void*>(arr_cast.get_data()) ==
            reinterpret_cast<const void*>(data));
    REQUIRE(arr_cast.has_mutable_data() == true);
    REQUIRE(reinterpret_cast<const void*>(arr_cast.get_mutable_data()) ==
            reinterpret_cast<const void*>(data));
    REQUIRE(arr_cast.get_count() == count * sizeof(float));
    REQUIRE(arr_cast.get_size() == arr.get_size());
}

TEST("reinterpret_cast with incompatible types throw exception") {
    constexpr std::int64_t count = 3;
    float data[count] = { -1.0, 2.0, -3.0 };
    const auto arr = array<float>::wrap(data, count);

    REQUIRE_THROWS_AS(detail::reinterpret_array_cast<double>(arr), invalid_argument);
}

TEST("cast mutable array to immutable") {
    constexpr std::int64_t count = 3;
    float data[count] = { -1.0, 2.0, -3.0 };
    const auto arr = array<float>::wrap(data, count);

    const auto immutable_array = detail::discard_mutable_data(arr);

    REQUIRE(arr.has_mutable_data() == true);
    REQUIRE(immutable_array.has_mutable_data() == false);
    REQUIRE(immutable_array.get_data() == arr.get_data());
}

#ifdef ONEDAL_DATA_PARALLEL
TEST("can reinterpret_cast immutable array with queue") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    constexpr std::int64_t count = 2;
    const float data[count] = { -1.0, 2.0 };
    const auto mat_device = la::matrix<float>::wrap(data, { count, 1 }).to_device(q);
    const auto arr = array<float>::wrap(q, mat_device.get_data(), count);

    const auto arr_cast = detail::reinterpret_array_cast<byte_t>(arr);

    REQUIRE(reinterpret_cast<const void*>(arr_cast.get_data()) ==
            reinterpret_cast<const void*>(mat_device.get_data()));
    REQUIRE(arr_cast.has_mutable_data() == false);
    REQUIRE(arr_cast.get_count() == count * sizeof(float));
    REQUIRE(arr_cast.get_size() == arr.get_size());
    REQUIRE(arr_cast.get_queue().value() == q);
}

TEST("can reinterpret_cast mutable array with queue") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    constexpr std::int64_t count = 2;
    const float data[count] = { -1.0, 2.0 };
    const auto mat_device = la::matrix<float>::wrap(data, { count, 1 }).to_device(q);
    const auto arr = array<float>::wrap(q, mat_device.get_mutable_data(), count);

    const auto arr_cast = detail::reinterpret_array_cast<byte_t>(arr);

    REQUIRE(reinterpret_cast<const void*>(arr_cast.get_data()) ==
            reinterpret_cast<const void*>(mat_device.get_data()));
    REQUIRE(arr_cast.has_mutable_data() == true);
    REQUIRE(reinterpret_cast<const void*>(arr_cast.get_mutable_data()) ==
            reinterpret_cast<const void*>(mat_device.get_data()));
    REQUIRE(arr_cast.get_count() == count * sizeof(float));
    REQUIRE(arr_cast.get_size() == arr.get_size());
    REQUIRE(arr_cast.get_queue().value() == q);
}
#endif

} // namespace oneapi::dal::test
