/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/algo/covariance/test/fixture.hpp"

namespace oneapi::dal::covariance::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace cov = oneapi::dal::covariance;

template <typename TestType>
class covariance_online_test : public covariance_test<TestType, covariance_online_test<TestType>> {
};

TEMPLATE_LIST_TEST_M(covariance_online_test,
                     "covariance  fill_normal common flow",
                     "[covariance][integration][online]",
                     covariance_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe input =
        GENERATE_DATAFRAME(te::dataframe_builder{ 100, 100 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 100, 100 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 250, 250 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 500, 100 }.fill_normal(0, 1, 7777));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto input_data_table_id = this->get_homogen_table_id();
    this->online_general_checks(input, input_data_table_id);
}

TEMPLATE_LIST_TEST_M(covariance_online_test,
                     "covariance fill_uniform common flow",
                     "[covariance][integration][online]",
                     covariance_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe input =
        GENERATE_DATAFRAME(te::dataframe_builder{ 1000, 20 }.fill_uniform(-30, 30, 7777),
                           te::dataframe_builder{ 100, 10 }.fill_uniform(0, 1, 7777),
                           te::dataframe_builder{ 100, 10 }.fill_uniform(-10, 10, 7777),
                           te::dataframe_builder{ 500, 40 }.fill_uniform(-100, 100, 7777),
                           te::dataframe_builder{ 500, 250 }.fill_uniform(0, 1, 7777));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto input_data_table_id = this->get_homogen_table_id();
    this->online_general_checks(input, input_data_table_id);
}

TEMPLATE_LIST_TEST_M(covariance_online_test,
                     "covariance fill_uniform nightly common flow",
                     "[covariance][integration][online][nightly]",
                     covariance_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe input =
        GENERATE_DATAFRAME(te::dataframe_builder{ 5000, 20 }.fill_uniform(-30, 30, 7777),
                           te::dataframe_builder{ 10000, 200 }.fill_uniform(-30, 30, 7777),
                           te::dataframe_builder{ 1000000, 20 }.fill_uniform(-0.5, 0.5, 7777));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto input_data_table_id = this->get_homogen_table_id();
    this->online_general_checks(input, input_data_table_id);
}

} // namespace oneapi::dal::covariance::test
