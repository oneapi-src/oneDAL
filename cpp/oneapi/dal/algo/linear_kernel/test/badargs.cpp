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

#include "oneapi/dal/algo/linear_kernel/compute.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::linear_kernel::test {

namespace te = dal::test::engine;

template <typename TestType>
class linear_kernel_badarg_test : public te::algo_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    static constexpr std::int64_t row_count_x = 5;
    static constexpr std::int64_t row_count_y = 3;
    static constexpr std::int64_t column_count = 4;
    static constexpr std::int64_t element_count_x = row_count_x * column_count;
    static constexpr std::int64_t element_count_y = row_count_y * column_count;

    auto get_descriptor() const {
        return linear_kernel::descriptor<float, Method, linear_kernel::task::compute>{};
    }

    table get_x_data(std::int64_t override_row_count = row_count_x,
                     std::int64_t override_column_count = column_count) const {
        ONEDAL_ASSERT(override_row_count * override_column_count <= element_count_x);
        return homogen_table::wrap(x_data_.data(), override_row_count, override_column_count);
    }

    table get_y_data(std::int64_t override_row_count = row_count_y,
                     std::int64_t override_column_count = column_count) const {
        ONEDAL_ASSERT(override_row_count * override_column_count <= element_count_y);
        return homogen_table::wrap(y_data_.data(), override_row_count, override_column_count);
    }

private:
    static constexpr std::array<float, element_count_x> x_data_ = { 1.0,  1.0,  2.0,  2.0,  1.0,
                                                                    2.0,  2.0,  1.0,  -1.0, -1.0,
                                                                    -1.0, -2.0, -2.0, -1.0, -2.0,
                                                                    -2.0, -1.0, -2.0, 1.0,  2.0 };

    static constexpr std::array<float, element_count_y> y_data_ = { 1.0,  1.0,  2.0,  2.0,
                                                                    1.0,  2.0,  -1.0, -2.0,
                                                                    -2.0, -1.0, -2.0, -2.0 };
};

using lr_kernel_types = COMBINE_TYPES((float, double), (linear_kernel::method::dense));

#define LINEAR_KERNEL_BADARG_TEST(name)             \
    TEMPLATE_LIST_TEST_M(linear_kernel_badarg_test, \
                         name,                      \
                         "[linear_kernel][badarg]", \
                         lr_kernel_types)

LINEAR_KERNEL_BADARG_TEST("throws if x data is empty") {
    const auto linear_kernel_desc = this->get_descriptor();

    REQUIRE_THROWS_AS(this->compute(linear_kernel_desc, homogen_table{}, this->get_y_data()),
                      domain_error);
}

LINEAR_KERNEL_BADARG_TEST("throws if y data is empty") {
    const auto linear_kernel_desc = this->get_descriptor();

    REQUIRE_THROWS_AS(this->compute(linear_kernel_desc, this->get_x_data(), homogen_table{}),
                      domain_error);
}

LINEAR_KERNEL_BADARG_TEST("throws if x columns count neq y columns count") {
    const auto linear_kernel_desc = this->get_descriptor();

    REQUIRE_THROWS_AS(this->compute(linear_kernel_desc,
                                    this->get_x_data(this->row_count_x, 3),
                                    this->get_y_data()),
                      invalid_argument);
}

} // namespace oneapi::dal::linear_kernel::test
