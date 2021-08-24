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

#include "oneapi/dal/algo/basic_statistics/compute.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::basic_statistics::test {

namespace te = dal::test::engine;

template <typename TestType>
class bs_badarg_test : public te::algo_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    auto get_descriptor() const {
        return basic_statistics::descriptor<Float, Method>{};
    }
};

using bs_types = COMBINE_TYPES((float, double), (basic_statistics::method::dense));

#define BS_BADARG_TEST(name) \
    TEMPLATE_LIST_TEST_M(bs_badarg_test, name, "[basic_statistics_kernel][badarg]", bs_types)

BS_BADARG_TEST("throws if data is empty") {
    const auto desc = this->get_descriptor();

    REQUIRE_THROWS_AS(this->compute(desc, homogen_table{}), domain_error);
}

} // namespace oneapi::dal::basic_statistics::test
