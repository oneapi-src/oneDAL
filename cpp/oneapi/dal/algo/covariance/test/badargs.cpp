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

#include "oneapi/dal/algo/covariance/compute.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::covariance::test {

namespace te = dal::test::engine;

template <typename Method>
class covariance_badarg_test : public te::algo_fixture {
public:
    static constexpr std::int64_t row_count = 5;
    static constexpr std::int64_t component_count = 4;
    static constexpr std::int64_t element_count = row_count * component_count;

    auto get_descriptor() const {
        return covariance::descriptor<float, Method, covariance::task::compute>{};
    }

    table get_input_data(std::int64_t override_row_count = row_count,
                         std::int64_t override_component_count = component_count) const {
        ONEDAL_ASSERT(override_row_count * override_component_count <= element_count);
        return homogen_table::wrap(data_.data(), override_row_count, override_component_count);
    }

private:
    static constexpr std::array<float, element_count> data_ = { 1.0,  1.0,  2.0,  2.0,  1.0,
                                                                2.0,  2.0,  1.0,  -1.0, -1.0,
                                                                -1.0, -2.0, -2.0, -1.0, -2.0,
                                                                -2.0, -1.0, -2.0, 1.0,  2.0 };
};

#define COVARIANCE_BADARG_TEST(name) \
    TEMPLATE_TEST_M(covariance_badarg_test, name, "[covariance][badarg]", covariance::method::dense)

COVARIANCE_BADARG_TEST("throws if input data is empty") {
    const auto covariance_desc = this->get_descriptor();

    REQUIRE_THROWS_AS(this->compute(covariance_desc, homogen_table{}), domain_error);
}

} // namespace oneapi::dal::covariance::test
