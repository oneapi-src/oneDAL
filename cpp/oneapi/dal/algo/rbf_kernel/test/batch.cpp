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

#include "oneapi/dal/algo/rbf_kernel/test/fixture.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::rbf_kernel::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

template <typename TestType>
class rbf_kernel_batch_test : public rbf_kernel_test<TestType, rbf_kernel_batch_test<TestType>> {};

TEMPLATE_LIST_TEST_M(rbf_kernel_batch_test,
                     "rbf_kernel common flow",
                     "[rbf_kernel][integration][batch]",
                     rbf_kernel_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe x_data =
        GENERATE_DATAFRAME(//te::dataframe_builder{ 50, 50 }.fill_normal(0, 1, 7777),
                           //te::dataframe_builder{ 100, 50 }.fill_normal(0, 1, 7777),
                           //te::dataframe_builder{ 250, 50 }.fill_normal(0, 1, 7777),
                           te::dataframe_builder{ 3, 3 }.fill_normal(0, 1, 7777));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto x_data_table_id = this->get_homogen_table_id();

    const te::dataframe y_data =
        GENERATE_DATAFRAME(//te::dataframe_builder{ 50, 50 }.fill_normal(0, 1, 7777),
                           //te::dataframe_builder{ 100, 50 }.fill_normal(0, 1, 8888),
                           //te::dataframe_builder{ 200, 50 }.fill_normal(0, 1, 8888),
                           te::dataframe_builder{ 3, 3 }.fill_normal(0, 1, 8888));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto y_data_table_id = this->get_homogen_table_id();

    const double sigma = GENERATE_COPY(0.8, 1.0, 5.0);

    this->general_checks(x_data, y_data, sigma, x_data_table_id, y_data_table_id);
}

TEMPLATE_LIST_TEST_M(rbf_kernel_batch_test,
                     "rbf_kernel compute one element matrix",
                     "[rbf_kernel][integration][batch]",
                     rbf_kernel_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe x_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 1, 1 }.fill_normal(0, 1, 7777));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto x_data_table_id = this->get_homogen_table_id();

    const te::dataframe y_data =
        GENERATE_DATAFRAME(te::dataframe_builder{ 1, 1 }.fill_normal(0, 1, 8888));

    // Homogen floating point type is the same as algorithm's floating point type
    const auto y_data_table_id = this->get_homogen_table_id();

    const double sigma = GENERATE_COPY(0.8, 1.0, 5.0);

    this->general_checks(x_data, y_data, sigma, x_data_table_id, y_data_table_id);
}

} // namespace oneapi::dal::rbf_kernel::test
