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
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/serialization.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

template <typename T>
class array_serialization_test : te::policy_fixture {
public:
    array<T> get_empty_array() {
        return array<T>{};
    }

    array<T> get_host_backed_array(std::int64_t count, int seed = 7777) {
        auto random_mat = la::generate_uniform_matrix<T>({ 1, count }, -10, 10, seed);
        return random_mat.get_array();
    }

#ifdef ONEDAL_DATA_PARALLEL
    array<T> get_device_backed_array(std::int64_t count, int seed = 7777) {
        auto random_mat = la::generate_uniform_matrix<T>({ 1, count }, -10, 10, seed);
        return random_mat.to_device(this->get_queue()).get_array();
    }
#endif

    void check_array_invariants(const array<T>& deserialized) {
        if (deserialized.get_count() > 0) {
            // We assume deserialized always has mutable data
            REQUIRE(deserialized.has_mutable_data());
            REQUIRE(deserialized.get_data() != nullptr);
            REQUIRE(deserialized.get_mutable_data() != nullptr);
            REQUIRE(deserialized.get_mutable_data() == deserialized.get_data());
        }
    }

    void compare_arrays(const array<T>& original, const array<T>& deserialized) {
        REQUIRE(original.get_count() == deserialized.get_count());

        if (original.get_count() > 0) {
            const auto original_host = la::matrix<T>::wrap(original).to_host().get_array();
            const auto deserialized_host = la::matrix<T>::wrap(deserialized).to_host().get_array();

            for (std::int64_t i = 0; i < original.get_count(); i++) {
                REQUIRE(deserialized_host[i] == original_host[i]);
            }
        }
    }
};

using array_types = std::tuple<std::int32_t, float, double>;

TEMPLATE_LIST_TEST_M(array_serialization_test,
                     "serialize/deserialize empty array",
                     "[empty]",
                     array_types) {
    const auto empty_array = this->get_empty_array();

    const auto deserialized = te::serialize_deserialize(empty_array);

    this->check_array_invariants(deserialized);
    this->compare_arrays(empty_array, deserialized);
}

TEMPLATE_LIST_TEST_M(array_serialization_test,
                     "serialize/deserialize host array",
                     "[host]",
                     array_types) {
    const std::int64_t count = GENERATE(1, 10, 1000);
    const auto host_array = this->get_host_backed_array(count);

    const auto deserialized = te::serialize_deserialize(host_array);

    this->check_array_invariants(deserialized);
    this->compare_arrays(host_array, deserialized);
}

#ifdef ONEDAL_DATA_PARALLEL
TEMPLATE_LIST_TEST_M(array_serialization_test,
                     "serialize device array, deserialize to host array",
                     "[device]",
                     array_types) {
    const std::int64_t count = GENERATE(1, 10, 1000);
    const auto device_array = this->get_device_backed_array(count);

    const auto deserialized = te::serialize_deserialize(device_array);

    this->check_array_invariants(deserialized);
    this->compare_arrays(device_array, deserialized);
}
#endif

TEMPLATE_TEST_M(array_serialization_test,
                "deserialize array to wrong type",
                "[badarg]",
                float,
                double) {
    const std::int64_t count = 10;
    const auto host_array = this->get_host_backed_array(count);

    array<std::int32_t> deserialized;
    REQUIRE_THROWS_AS(te::serialize_deserialize(host_array, deserialized), invalid_argument);
}

} // namespace oneapi::dal::test
