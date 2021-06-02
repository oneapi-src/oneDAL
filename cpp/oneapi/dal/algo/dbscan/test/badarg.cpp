/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <array>

#include "oneapi/dal/algo/dbscan/compute.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::dbscan::test {

namespace te = dal::test::engine;

template <typename Method>
class dbscan_badarg_test : public te::algo_fixture {
public:
    static constexpr std::int64_t row_count = 8;
    static constexpr std::int64_t column_count = 2;
    static constexpr std::int64_t element_count = row_count * column_count;

    auto get_descriptor() const {
        return dbscan::descriptor<float, Method>(1.0, 2);
    }

    table get_compute_data(std::int64_t override_row_count = row_count,
                           std::int64_t override_column_count = column_count) const {
        ONEDAL_ASSERT(override_row_count * override_column_count <= element_count);
        return homogen_table::wrap(compute_data_.data(), override_row_count, override_column_count);
    }

    table get_weights(std::int64_t override_row_count = row_count) const {
        ONEDAL_ASSERT(override_row_count <= element_count);
        return homogen_table::wrap(weights_.data(), override_row_count, 1);
    }

private:
    static constexpr std::array<float, element_count> compute_data_ = {
        1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0
    };

    static constexpr std::array<float, row_count> weights_ = { 1.0, 1.0, 2.0, 2.0,
                                                               1.0, 2.0, 2.0, 1.0 };
};

#define DBSCAN_BADARG_TEST(name) \
    TEMPLATE_TEST_M(dbscan_badarg_test, name, "[dbscan][badarg]", method::brute_force)

DBSCAN_BADARG_TEST("accepts positive min observations") {
    REQUIRE_NOTHROW(this->get_descriptor().set_min_observations(1));
}

DBSCAN_BADARG_TEST("throws if epsilon is negative") {
    REQUIRE_THROWS_AS(this->get_descriptor().set_epsilon(-1.0), domain_error);
}

DBSCAN_BADARG_TEST("accepts positive epsilon") {
    REQUIRE_NOTHROW(this->get_descriptor().set_epsilon(1.0));
}

DBSCAN_BADARG_TEST("throws if min_observatons is negative") {
    REQUIRE_THROWS_AS(this->get_descriptor().set_min_observations(-1), domain_error);
}

} // namespace oneapi::dal::dbscan::test
