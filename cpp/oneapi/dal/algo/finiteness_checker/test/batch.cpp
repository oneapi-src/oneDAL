/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include <limits>
#include <cmath>

#include "oneapi/dal/algo/finiteness_checker/compute.hpp"

#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::finiteness_checker::test {

namespace te = dal::test::engine;

template <typename TestType>
class finite_checker_batch_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    void check_finiteness(const te::dataframe& x_data,
                          bool allowNaN,
                          double value,
                          const te::table_id& x_data_table_id) {
        CAPTURE(allowNaN);
        const table x = x_data.get_table(this->get_policy(), x_data_table_id);

        INFO("create descriptor");
        const auto finiteness_desc =
            finiteness_checker::descriptor<Float, Method>{}.set_allow_NaN(allowNaN);

        INFO("run compute");
        const auto compute_result = this->compute(finiteness_desc, x);
        if (compute_result != (std::isinf(value) || std::isnan(value) && allowNaN)) {
            CAPTURE(compute_result, value, allowNaN);
            FAIL();
        }
        SUCCEED();
    }
};

using finiteness_types = COMBINE_TYPES((float, double), (finiteness_checker::method::dense));

TEMPLATE_LIST_TEST_M(finiteness_checker_batch_test,
                     "finiteness checker typical",
                     "[finiteness_checker][integration][batch]",
                     finiteness_types) {
    SKIP_IF(this->not_float64_friendly());

    // Initialize values
    const te::dataframe x_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 50, 50 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 100, 50 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 250, 50 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 1100, 50 }.fill_normal(0, 1, 7777));
    auto x_data_mutable = x_data.get_array().get_mutable_data();
    const double value = GENERATE(0.0,
                           -std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN());
    const bool allowNaN = GENERATE(0, 1);
    x_data_mutable[45] = value;

    // Homogen floating point type is the same as algorithm's floating point type
    const auto x_data_table_id = this->get_homogen_table_id();

    this->check_finiteness(x_data, allowNaN, value, x_data_table_id);
}

TEMPLATE_LIST_TEST_M(finiteness_checker_batch_test,
                     "finiteness_checker compute one element matrix",
                     "[finiteness_checker][integration][batch]",
                     finiteness_types) {
    SKIP_IF(this->not_float64_friendly());

    // Initialize values to doubles
    const double value = GENERATE(0.0,
                           -std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN());
    const bool allowNaN = GENERATE(0, 1);

    const te::dataframe x_data = GENERATE_DATAFRAME(te::dataframe_builder{ 1, 1 }.fill(value));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto x_data_table_id = this->get_homogen_table_id();

    this->check_finiteness(x_data, allowNaN, value, x_data_table_id);
}

} // namespace oneapi::dal::finiteness_checker::test
