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

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/detail/serialization.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/serialization.hpp"

namespace oneapi::dal::test {

namespace te = dal::test::engine;

TEMPLATE_TEST("can write to binary_ouput_archive",
              "[binary_ouput_archive]",
              float,
              double,
              std::int32_t) {
    detail::binary_output_archive archive;

    SECTION("single value") {
        const TestType original = TestType(3.14);
        archive(&original, detail::make_data_type<TestType>());

        const TestType written = *reinterpret_cast<const TestType*>(archive.get_data());
        REQUIRE(written == original);
    }

    SECTION("multiple values") {
        constexpr std::int64_t count = 10;
        TestType original[count];
        for (std::int64_t i = 0; i < count; i++) {
            original[i] = TestType(i);
        }

        archive(original, detail::make_data_type<TestType>(), count);

        const TestType* written = reinterpret_cast<const TestType*>(archive.get_data());
        for (std::int64_t i = 0; i < count; i++) {
            REQUIRE(written[i] == TestType(i));
        }
    }
}

TEMPLATE_TEST("can read from binary_input_archive",
              "[binary_input_archive]",
              float,
              double,
              std::int32_t) {
    SECTION("singe value") {
        const TestType original = TestType(3.14);

        detail::binary_input_archive archive{ reinterpret_cast<const byte_t*>(&original),
                                              sizeof(TestType) };

        TestType read;
        archive(&read, detail::make_data_type<TestType>());

        REQUIRE(read == original);
    }

    SECTION("multiple values") {
        constexpr std::int64_t count = 10;
        TestType original[count];
        for (std::int64_t i = 0; i < count; i++) {
            original[i] = TestType(i);
        }

        detail::binary_input_archive archive{ reinterpret_cast<const byte_t*>(original),
                                              sizeof(TestType) * count };

        TestType read[count];
        archive(read, detail::make_data_type<TestType>(), count);

        for (std::int64_t i = 0; i < count; i++) {
            REQUIRE(read[i] == TestType(i));
        }
    }
}

TEMPLATE_TEST("serialize/deserialize array to binary archive",
              "[array]",
              float,
              double,
              std::int32_t) {
    const std::int64_t count = 100;
    const auto original = array<TestType>::empty(count);
    for (std::int64_t i = 0; i < count; i++) {
        original.get_mutable_data()[i] = i;
    }

    INFO("serialize");
    detail::binary_output_archive output_archive;
    detail::serialize(original, output_archive);

    INFO("deserialize");
    array<TestType> deserialized;
    detail::binary_input_archive input_archive{ output_archive.get_data(),
                                                output_archive.get_size() };
    detail::deserialize(deserialized, input_archive);

    REQUIRE(deserialized.get_count() == original.get_count());
    for (std::int64_t i = 0; i < count; i++) {
        REQUIRE(deserialized[i] == TestType(i));
    }
}

TEST("binary_input_archive throws if truncated data buffer is provided", "[binary_input_archive]") {
    const std::int64_t count = 100;
    const auto original = array<float>::empty(count);
    for (std::int64_t i = 0; i < count; i++) {
        original.get_mutable_data()[i] = float(i);
    }

    INFO("serialize");
    detail::binary_output_archive output_archive;
    detail::serialize(original, output_archive);

    INFO("deserialize");
    array<float> deserialized;
    detail::binary_input_archive input_archive{ output_archive.get_data(),
                                                output_archive.get_size() / 2 };
    REQUIRE_THROWS_AS(detail::deserialize(deserialized, input_archive), invalid_argument);
}

} // namespace oneapi::dal::test
