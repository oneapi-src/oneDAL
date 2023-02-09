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

#pragma once

#include "oneapi/dal/algo/objective_function/compute.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::objective_function::test {

namespace pr = dal::backend::primitives;
namespace te = dal::test::engine;
namespace de = dal::detail;
namespace obj_fun = oneapi::dal::objective_function;
namespace lg = oneapi::dal::logloss_objective;

template <typename TestType, typename Derived>
class logloss_test : public te::crtp_algo_fixture<TestType, Derived> {
// te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
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

    void gen_input() {
        const te::dataframe data_df = GENERATE_DATAFRAME(te::dataframe_builder{ n_, p_ }.fill_normal(-0.5, 0.5, 7777));
        const te::dataframe params_df = GENERATE_DATAFRAME(te::dataframe_builder{ 1, p_ + 1 }.fill_normal(-0.5, 0.5, 7777));
        auto resp = array<std::int32_t>::zeros(n_);
        auto ptr = resp.get_mutable_data();
        srand(2007 + n_ + p_ + n_ * p_);
        for (std::int64_t i = 0; i < n_; ++i) {
            ptr[i] = rand() % 2;
        }

        responses_ = de::homogen_table_builder{}.reset(resp, 1, n_).build();
        data_ = data_df.get_table(this->get_policy(), this->get_homogen_table_id());
        params_ = params_df.get_table(this->get_policy(), this->get_homogen_table_id());
    }

    void general_checks(const double L1 = 0, const double L2 = 0) {


        INFO("create descriptor value");
        auto desc = get_descriptor(obj_fun::result_options::value, L1, L2);
        INFO("run compute value");
        auto compute_result = this->compute(desc, data_, params_, responses_);
        
        INFO("create descriptor gradient");
        desc = get_descriptor(obj_fun::result_options::gradient, L1, L2);
        INFO("run compute gradient");
        compute_result = this->compute(desc, data_, params_, responses_);
        
        INFO("create descriptor hessian");
        desc = get_descriptor(obj_fun::result_options::hessian, L1, L2);
        INFO("run compute hessian");
        compute_result = this->compute(desc, data_, params_, responses_);


        INFO("create descriptor value + gradient + hessian");
        desc = get_descriptor(obj_fun::result_options::value | 
        obj_fun::result_options::gradient | obj_fun::result_options::hessian, L1, L2);
        INFO("run compute value + gradient + hessian");
        compute_result = this->compute(desc, data_, params_, responses_);
        check_compute_result(L1, L2, compute_result);

    }

    void check_compute_result(Float L1, Float L2,
                              const objective_function::compute_result<>& result) {
        auto data_arr = row_accessor<const Float>(data_).pull({ 0, -1 });
        auto params_arr = row_accessor<const Float>(params_).pull({ 0, -1 });
        auto resp_arr = row_accessor<const Float>(responses_).pull({ 0, -1 });

        // std::int64_t n_ = data.get_row_count();
        // std::int64_t p_ = data.get_column_count();

        auto pred_arr = array<float_t>::zeros(n_);
        auto pred_ptr = pred_arr.get_mutable_data();

        for (std::int64_t i = 0; i < n_; ++i) {
            for (std::int64_t j = 0; j < p_; ++j) {
                pred_ptr[i] += data_arr[i * p_ + j] * params_arr[j + 1];
            }
            pred_ptr[i] += params_arr[0];
            pred_ptr[i] = 1 / (1 + std::exp(-pred_ptr[i]));
        }
        const double tol = te::get_tolerance<Float>(1e-4, 1e-6);

        if (result.get_result_options().test(result_options::value)) {
            const auto value = result.get_value();
            REQUIRE(value.get_row_count() == 1);
            REQUIRE(value.get_column_count() == 1);
            REQUIRE(te::has_no_nans(value));
            auto val_arr = row_accessor<const Float>(value).pull({ 0, -1 });
            Float val = val_arr[0];
            Float ans = 0;
            for (std::int64_t i = 0; i < n_; ++i) {

                ans -= resp_arr[i] * std::log(pred_ptr[i]) + (1 - resp_arr[i]) * std::log(1 - pred_ptr[i]);
            }
            for (std::int64_t i = 0; i <= p_; ++i) {
                ans += L1 * std::abs(params_arr[i]) + L2 * params_arr[i] * params_arr[i];
            }
            const double diff = std::abs(val - ans);
            CHECK(diff < tol);
        }
        if (result.get_result_options().test(result_options::gradient)) {
            const auto gradient = result.get_gradient();
            REQUIRE(gradient.get_row_count() == 1);
            REQUIRE(gradient.get_column_count() == p_ + 1);
            auto grad_arr = row_accessor<const Float>(gradient).pull({ 0, -1 });
            for (std::int64_t j = 0; j <= p_; ++j) {
                Float ans = 0;
                for (std::int64_t i = 0; i < n_; ++i) {
                    Float x1 = j == 0 ? 1 : data_arr[i * p_ + j - 1];
                    ans += (pred_ptr[i] - resp_arr[i]) * x1;
                }
                ans += std::copysign(L1, params_arr[j]) + L2 * 2 * params_arr[j];
                const double diff = std::abs(grad_arr[j] - ans);
                // std::cout << j << ": " << grad_arr[j] << " " << ans << std::endl; 
                CHECK(diff < tol);
            }
        }

        if (result.get_result_options().test(result_options::hessian)) {
            const auto hessian = result.get_hessian();
            REQUIRE(hessian.get_row_count() == p_ + 1);
            REQUIRE(hessian.get_column_count() == p_ + 1);
            auto hess_arr = row_accessor<const Float>(hessian).pull({ 0, -1 });
            for (std::int64_t k = 0; k <= p_; ++k) {
                for (std::int64_t j = 0; j <= p_; ++j) {
                    Float ans = 0;
                    for (std::int64_t i = 0; i < n_; ++i) {
                        Float x1 = k == 0 ? 1 : data_arr[i * p_ + k - 1];
                        Float x2 = j == 0 ? 1 : data_arr[i * p_ + j - 1];
                        ans += x1 * x2 * pred_ptr[i] * (1 - pred_ptr[i]);
                    }
                    ans += (k == j) ? L2 * 2 : 0;
                    const double diff = std::abs(hess_arr[k * (p_ + 1) + j] - ans);
                    CHECK(diff < tol);
                }
            }
        }
        
    }

protected:
    std::int64_t n_ = 20;
    std::int64_t p_ = 10;
    table data_;
    table params_;
    table responses_;

};

using logloss_types = COMBINE_TYPES((float, double), (obj_fun::method::dense));

} // namespace oneapi::dal::objective_function::test