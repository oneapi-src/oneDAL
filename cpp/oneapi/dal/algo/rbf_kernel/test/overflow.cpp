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

#include "oneapi/dal/algo/rbf_kernel/compute.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/mocks.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::rbf_kernel::test {

namespace te = dal::test::engine;

template <typename TestType>
class rbf_kernel_overflow_test : public te::algo_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    static constexpr std::int64_t row_count_x = 0x7FFFFFFFF;
    static constexpr std::int64_t row_count_y = 0x7FFFFFFFF;
    static constexpr std::int64_t column_count = 2;

    auto get_descriptor() const {
        return rbf_kernel::descriptor<float, Method, rbf_kernel::task::compute>{};
    }

    table get_x_data() const {
        return te::dummy_table{ row_count_x, column_count };
    }

    table get_y_data() const {
        return te::dummy_table{ row_count_y, column_count };
    }
};

using rbf_kernel_types = COMBINE_TYPES((float, double), (rbf_kernel::method::dense));

#define RBF_KERNEL_OVERFLOW_TEST(name) \
    TEMPLATE_LIST_TEST_M(rbf_kernel_overflow_test, name, "[rbf_kernel][overflow]", rbf_kernel_types)

RBF_KERNEL_OVERFLOW_TEST("compute throws if result values table leads to overflow") {
    const auto rbf_kernel_desc = this->get_descriptor();

    REQUIRE_THROWS_AS(this->compute(rbf_kernel_desc, this->get_x_data(), this->get_y_data()),
                      range_error);
}

} // namespace oneapi::dal::rbf_kernel::test
