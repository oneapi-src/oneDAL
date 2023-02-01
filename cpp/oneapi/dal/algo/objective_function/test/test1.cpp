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

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::objective_function::test {

namespace pr = dal::backend::primitives;
namespace te = dal::test::engine;
namespace de = dal::detail;
namespace obj_fun = oneapi::dal::objective_function;
namespace lg = oneapi::dal::logloss_objective;

template <typename TestType>
class logloss_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    using input_t = obj_fun::compute_input<>;
    using result_t = obj_fun::compute_result<>;
    using descriptor_t = obj_fun::descriptor<Float, Method>;
    using objective_t = lg::descriptor<Float>;

    auto get_descriptor(obj_fun::result_option_id compute_mode, double L1 = 0, double L2 = 0) const {
        return descriptor_t(objective_t{L1, L2}).set_result_options(compute_mode);
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<Float>();
    }

    void general_checks(const te::dataframe& input_data, 
                        const te::dataframe& input_params, 
                        const table& responses, const te::table_id& data_table_id,
                        const obj_fun::result_option_id& compute_mode, double L1 = 0, double L2 = 0) {
        const table data = input_data.get_table(this->get_policy(), data_table_id);
        const table params = input_params.get_table(this->get_policy(), data_table_id);
        //const table responses = input_responses.get_table(this->get_policy(), data_table_id);

        INFO("create descriptor hessian");
        auto desc = get_descriptor(obj_fun::result_options::hessian, L1, L2);

        INFO("run compute hessian");
        auto compute_result = this->compute(desc, data, params, responses);
        // check_compute_result(data, compute_result);

    }

    //void check_compute_result(const table& data, const covariance::compute_result<>& result) {
        
    // }

};

using logloss_types = COMBINE_TYPES((float, double), (obj_fun::method::dense));

TEMPLATE_LIST_TEST_M(logloss_test,
                     "logloss hessian",
                     "[logloss][integration][gpu]",
                     logloss_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

    const std::int64_t n = 20;
    const std::int64_t p = 10; 
    auto q = this->get_queue();

    const te::dataframe data = GENERATE_DATAFRAME(te::dataframe_builder{ n, p }.fill_normal(-0.5, 0.5, 7777));
    const te::dataframe params = GENERATE_DATAFRAME(te::dataframe_builder{ 1, p + 1 }.fill_normal(-0.5, 0.5, 7777));
    auto resp = array<std::int32_t>::zeros(n);
    // auto resp_ptr = resp.get_mutable_data();
    const table responses = de::homogen_table_builder{}.reset(resp, 1, n).build();
    //const te::dataframe responses = homogen_table::wrap(resp_gpu.flatten(this->get_queue(), {}), 1, 20);

    // const obj_fun::result_option_id compute_mode = obj::fun::result_optionmhessian;

    const obj_fun::result_option_id compute_mode = obj_fun::result_options::hessian;

    const auto data_table_id = this->get_homogen_table_id();

    this->general_checks(data, params, responses, data_table_id, compute_mode, 1.1, 2.3);
}

} // namespace oneapi::dal::covariance::test
