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
                     "covariance common flow",
                     "[covariance][integration][online]",
                     covariance_types) {
    SKIP_IF(this->not_float64_friendly());

    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    const int64_t nBlocks = GENERATE(1, 3, 10);
    INFO("nBlocks=" << nBlocks);
    const bool assume_centered = GENERATE(true, false);
    INFO("assume_centered=" << assume_centered);
    const bool bias = GENERATE(true, false);
    INFO("bias=" << bias);
    const cov::result_option_id result_option =
        GENERATE(covariance::result_options::means,
                 covariance::result_options::cov_matrix,
                 covariance::result_options::cor_matrix,
                 covariance::result_options::cor_matrix | covariance::result_options::cov_matrix,
                 covariance::result_options::cor_matrix | covariance::result_options::cov_matrix |
                     covariance::result_options::means);
    INFO("result_option=" << result_option);

    auto cov_desc = covariance::descriptor<Float, Method, covariance::task::compute>()
                        .set_result_options(result_option)
                        .set_assume_centered(assume_centered)
                        .set_bias(bias);

    const te::dataframe input =
        GENERATE_DATAFRAME(te::dataframe_builder{ 100, 100 }.fill_normal(0, 1, 7777),
                       te::dataframe_builder{ 500, 100 }.fill_normal(0, 1, 7777),
                       te::dataframe_builder{ 10000, 200 }.fill_uniform(-30, 30, 7777));

    INFO("num_rows=" << input.get_row_count());
    INFO("num_columns=" << input.get_column_count());

    // Homogen floating point type is the same as algorithm's floating point type
    const auto input_data_table_id = this->get_homogen_table_id();
    this->online_general_checks(input, input_data_table_id, nBlocks, cov_desc);
}

} // namespace oneapi::dal::covariance::test
