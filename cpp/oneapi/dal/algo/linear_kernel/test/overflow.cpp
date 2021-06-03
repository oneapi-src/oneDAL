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

#include "oneapi/dal/algo/linear_kernel/compute.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/mocks.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::linear_kernel::test {

namespace te = dal::test::engine;

template <typename Method>
class linear_kernel_overflow_test : public te::algo_fixture {
public:
    static constexpr std::int64_t row_count_x = 0x7FFFFFFFF;
    static constexpr std::int64_t row_count_y = 0x7FFFFFFFF;
    static constexpr std::int64_t column_count = 2;

    auto get_descriptor() const {
        return linear_kernel::descriptor<float, Method, linear_kernel::task::compute>{};
    }

    table get_x_data() const {
        return te::dummy_table{ row_count_x, column_count };
    }

    table get_y_data() const {
        return te::dummy_table{ row_count_y, column_count };
    }
};

#define LINEAR_KERNEL_OVERFLOW_TEST(name)        \
    TEMPLATE_TEST_M(linear_kernel_overflow_test, \
                    name,                        \
                    "[linear_kernel][overflow]", \
                    linear_kernel::method::dense)

LINEAR_KERNEL_OVERFLOW_TEST("compute throws if result values table leads to overflow") {
    const auto linear_kernel_desc = this->get_descriptor();

    REQUIRE_THROWS_AS(this->compute(linear_kernel_desc, this->get_x_data(), this->get_y_data()),
                      range_error);
}

} // namespace oneapi::dal::linear_kernel::test
