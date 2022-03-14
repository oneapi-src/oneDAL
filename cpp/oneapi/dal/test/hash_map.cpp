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

#include <string>
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/detail/hash_map.hpp"

namespace oneapi::dal::test {

using test_hash_map_t = detail::hash_map<std::string, std::string>;

TEST("can create hash_map with capacity") {
    const std::int64_t capacity = 100;
    test_hash_map_t map{ capacity };
}

TEST("hash_map constructor throws if non-positive capacity is given") {
    const std::int64_t capacity = GENERATE(0, -1);
    CAPTURE(capacity);
    REQUIRE_THROWS_AS(test_hash_map_t{ capacity }, invalid_argument);
}

TEST("can insert single key to hash_map", "[insert_unique]") {
    const std::int64_t capacity = 100;
    test_hash_map_t map{ capacity };

    map.set("key", "value");

    REQUIRE(map.has("key"));
    REQUIRE(map.get("key") == "value");
}

TEST("can insert multiple unique keys to hash_map when "
     "number of keys are smaller than capacity",
     "[insert_unique]") {
    const std::int64_t count = 10;
    const std::int64_t capacity = count * 1000;
    test_hash_map_t map{ capacity };

    for (std::int64_t i = 0; i < count; i++) {
        map.set(fmt::format("key_{}", i), fmt::format("value_{}", i));
    }

    for (std::int64_t i = 0; i < count; i++) {
        const auto key = fmt::format("key_{}", i);
        const auto value = fmt::format("value_{}", i);
        CAPTURE(key, value);
        REQUIRE(map.has(key));
        REQUIRE(map.get(key) == value);
    }
}

TEST("can insert multiple unique keys to hash_map when "
     "number of keys are larger than capacity",
     "[insert_unique]") {
    const std::int64_t count = 1000;
    const std::int64_t capacity = count / 100;
    test_hash_map_t map{ capacity };

    for (std::int64_t i = 0; i < count; i++) {
        map.set(fmt::format("key_{}", i), fmt::format("value_{}", i));
    }

    for (std::int64_t i = 0; i < count; i++) {
        const auto key = fmt::format("key_{}", i);
        const auto value = fmt::format("value_{}", i);
        CAPTURE(key, value);
        REQUIRE(map.has(key));
        REQUIRE(map.get(key) == value);
    }
}

TEST("can insert duplicated keys", "[insert_duplicates]") {
    const std::int64_t capacity = 100;
    test_hash_map_t map{ capacity };

    map.set("key", "value_1");
    map.set("key", "value_2");

    REQUIRE(map.has("key"));
    REQUIRE(map.get("key") == "value_2");
}

} // namespace oneapi::dal::test
