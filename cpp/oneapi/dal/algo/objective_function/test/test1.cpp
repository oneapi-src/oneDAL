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

// #pragma once

#include "oneapi/dal/algo/objective_function/compute.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::objective_function::test {

namespace te = dal::test::engine;
namespace obj_fun = oneapi::dal::objective_function;
namespace lgloss = oneapi::dal::logloss_objective;

template <typename TestType, typename Derived>
class logloss_test : public te::crtp_algo_fixture<TestType, Derived> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    using input_t = obj_fun::compute_input<>;
    using result_t = obj_fun::compute_result<>;
    using descriptor_t = obj_fun::descriptor<Float, Method>;
    using objective_t = lgloss::descriptor<Float>;

    auto get_descriptor(obj_fun::result_option_id compute_mode, double L1 = 0, double L2 = 0) const {
        return descriptor_t(objective_t{L1, L2}).set_result_options(compute_mode);
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<Float>();
    }

    void general_checks(const te::dataframe& input_data, 
                        const te::dataframe& input_params, 
                        const te::dataframe& intput_responses, const te::table_id& data_table_id) {
        const table data = input_data.get_table(this->get_policy(), data_table_id);
        const table params = input_params.get_table(this->get_policy(), data_table_id);
        const table responses = input_responses.get_table(this->get_policy(), data_table_id);

        INFO("create descriptor hessian")
        auto desc =
            obj_fun::descriptor<Float, Method, covariance::task::compute, lgloss::descriptor<Float>>().set_result_options(
                obj_fun::result_options::hessian);
        INFO("run compute hessian");
        auto compute_result = this->compute(desc, data, params, responses);
        // check_compute_result(data, compute_result);

    }

    //void check_compute_result(const table& data, const covariance::compute_result<>& result) {
        
    // }

};

using logloss_types = COMBINE_TYPES((float, double), (obj_fun::method::dense_batch));

TEMPLATE_LIST_TEST_M(logloss_test,
                     "logloss hessian",
                     "[logloss][integration][gpu]",
                     logloss_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe data = te::dataframe_builder{ 20, 10 }.fill_normal(-0.5, 0.5, 7777);
    const te::dataframe params = te::dataframe_builder{ 1, 11 }.fill_normal(-0.5, 0.5, 7777);
    const te::dataframe responses = te::dataframe_builder{ 1, 20 }.fill_normal(-0.5, 0.5, 7777);
    this->set_rank_count(GENERATE(2, 4));


    const cov::result_option_id compute_mode = GENERATE_COPY(mode_mean,
                                                             mode_cor,
                                                             mode_cov,
                                                             mode_cor_mean,
                                                             mode_cov_mean,
                                                             mode_cov_cor,
                                                             res_all);

    const auto data_table_id = this->get_homogen_table_id();

    this->general_checks(data, params, responses, data_table_id);
}

} // namespace oneapi::dal::covariance::test
