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
#include <random>

namespace oneapi::dal::objective_function::test {

namespace te = dal::test::engine;
namespace de = dal::detail;
namespace obj_fun = oneapi::dal::objective_function;
namespace lg = oneapi::dal::logloss_objective;

template <typename TestType, typename Derived>
class logloss_test : public te::crtp_algo_fixture<TestType, Derived> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    using input_t = obj_fun::compute_input<>;
    using result_t = obj_fun::compute_result<>;
    using descriptor_t = obj_fun::descriptor<Float, Method>;
    using objective_t = lg::descriptor<Float>;

    auto get_descriptor(obj_fun::result_option_id compute_mode) const {
        return descriptor_t(objective_t{ L1_, L2_, fit_intercept_ })
            .set_result_options(compute_mode);
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<Float>();
    }

    void set_reg_coefs(double L1, double L2) {
        this->L1_ = L1;
        this->L2_ = L2;
    }

    void set_intercept_flag(bool fit_intercept) {
        this->fit_intercept_ = fit_intercept;
    }

    void gen_input() {
        std::mt19937 rnd(2007 + n_ + p_ + n_ * p_);
        const te::dataframe data_df =
            GENERATE_DATAFRAME(te::dataframe_builder{ n_, p_ }.fill_normal(-0.5, 0.5, 7777));
        const te::dataframe params_df =
            GENERATE_DATAFRAME(te::dataframe_builder{ p_ + 1, 1 }.fill_normal(-0.5, 0.5, 7777));
        auto resp = array<std::int32_t>::zeros(n_);
        auto* const ptr = resp.get_mutable_data();
        std::generate(ptr, ptr + n_, [&]() {
            return rnd() % 2;
        });
        responses_ = de::homogen_table_builder{}.reset(resp, n_, 1).build();
        data_ = data_df.get_table(this->get_policy(), this->get_homogen_table_id());
        params_ = params_df.get_table(this->get_policy(), this->get_homogen_table_id());
    }

    void general_checks() {
        INFO("create descriptor value");
        auto desc = get_descriptor(obj_fun::result_options::value);
        INFO("run compute value");
        auto compute_result = this->compute(desc, data_, params_, responses_);
        check_compute_result(compute_result);

        INFO("create descriptor gradient");
        desc = get_descriptor(obj_fun::result_options::gradient);
        INFO("run compute gradient");
        compute_result = this->compute(desc, data_, params_, responses_);
        check_compute_result(compute_result);

        INFO("create descriptor value + gradient");
        desc = get_descriptor(obj_fun::result_options::value | obj_fun::result_options::gradient);
        INFO("run compute value + gradient");
        compute_result = this->compute(desc, data_, params_, responses_);
        check_compute_result(compute_result);

        INFO("create descriptor hessian");
        desc = get_descriptor(obj_fun::result_options::hessian);
        INFO("run compute hessian");
        compute_result = this->compute(desc, data_, params_, responses_);
        check_compute_result(compute_result);

        INFO("create descriptor value + hessian");
        desc = get_descriptor(obj_fun::result_options::value | obj_fun::result_options::hessian);
        INFO("run compute value + hessian");
        compute_result = this->compute(desc, data_, params_, responses_);
        check_compute_result(compute_result);

        INFO("create descriptor gradient + hessian");
        desc = get_descriptor(obj_fun::result_options::gradient | obj_fun::result_options::hessian);
        INFO("run compute gradient + hessian");
        compute_result = this->compute(desc, data_, params_, responses_);
        check_compute_result(compute_result);

        INFO("create descriptor value + gradient + hessian");
        desc = get_descriptor(obj_fun::result_options::value | obj_fun::result_options::gradient |
                              obj_fun::result_options::hessian);
        INFO("run compute value + gradient + hessian");
        compute_result = this->compute(desc, data_, params_, responses_);
        check_compute_result(compute_result);
    }

    void test_big() {
        auto desc =
            get_descriptor(obj_fun::result_options::value | obj_fun::result_options::gradient |
                           obj_fun::result_options::hessian);
        INFO("run compute");
        auto compute_result = this->compute(desc, data_, params_, responses_);
        stochastic_checks(compute_result);
    }

    void calculate_predictions(array<Float>& data_arr, array<Float>& params_arr, Float* pred_ptr) {
        const Float bottom = te::get_tolerance<Float>(1e-7, 1e-15);
        const Float top = Float(1.0) - bottom;
        for (std::int64_t i = 0; i < n_; ++i) {
            pred_ptr[i] = 0;
            for (std::int64_t j = 0; j < p_; ++j) {
                pred_ptr[i] += data_arr[i * p_ + j] * params_arr[j + 1];
            }
            if (fit_intercept_) {
                pred_ptr[i] += params_arr[0];
            }
            pred_ptr[i] = Float(1.0) / (Float(1.0) + std::exp(-pred_ptr[i]));
            if (pred_ptr[i] < bottom) {
                pred_ptr[i] = bottom;
            }
            if (pred_ptr[i] > top) {
                pred_ptr[i] = top;
            }
        }
    }

    void check_value(const objective_function::compute_result<>& result,
                     array<Float>& params_arr,
                     array<std::int32_t>& resp_arr,
                     const Float* pred_ptr,
                     const double tol = 1e-4) {
        if (result.get_result_options().test(result_options::value)) {
            const auto value = result.get_value();
            REQUIRE(value.get_row_count() == 1);
            REQUIRE(value.get_column_count() == 1);
            REQUIRE(te::has_no_nans(value));
            auto val_arr = row_accessor<const Float>(value).pull({ 0, -1 });
            const Float val = val_arr[0];
            Float ans = 0.0;
            for (std::int64_t i = 0; i < n_; ++i) {
                ans -= resp_arr[i] * std::log(pred_ptr[i]) +
                       (1 - resp_arr[i]) * std::log(1 - pred_ptr[i]);
            }
            ans /= n_;
            // We do not apply regularization to w_0
            for (std::int64_t i = 1; i <= p_; ++i) {
                ans += L1_ * std::abs(params_arr[i]) + L2_ * params_arr[i] * params_arr[i];
            }
            const double diff = std::abs(val - ans);
            REQUIRE(diff < tol);
        }
    }

    Float compute_gth_gradient_item(array<Float>& data_arr,
                                    array<Float>& params_arr,
                                    array<std::int32_t>& resp_arr,
                                    const Float* pred_ptr,
                                    std::int64_t j) {
        if (!fit_intercept_ && j == 0) {
            return 0.0;
        }
        Float ans = 0;
        for (std::int64_t i = 0; i < n_; ++i) {
            Float x1 = (j == 0) ? 1 : data_arr[i * p_ + j - 1];
            ans += (pred_ptr[i] - resp_arr[i]) * x1;
        }
        ans /= n_;
        // We do not apply regularization to w_0
        ans += j > 0 ? L2_ * 2 * params_arr[j] : 0;
        return ans;
    }

    void check_gradient(const objective_function::compute_result<>& result,
                        array<Float>& data_arr,
                        array<Float>& params_arr,
                        array<std::int32_t>& resp_arr,
                        const Float* pred_ptr,
                        const double tol = 1e-4,
                        const std::int32_t stochastic = 0) {
        std::mt19937 rnd(2007 + n_ + p_ + n_ * p_ + 1);
        if (result.get_result_options().test(result_options::gradient)) {
            const auto gradient = result.get_gradient();
            REQUIRE(gradient.get_row_count() == p_ + 1);
            REQUIRE(gradient.get_column_count() == 1);
            auto grad_arr = row_accessor<const Float>(gradient).pull({ 0, -1 });
            if (stochastic > 0) {
                for (std::int32_t num_checks = 0; num_checks < stochastic; ++num_checks) {
                    std::int64_t j = rnd() % (p_ + 1);
                    if (!fit_intercept_) {
                        j = rnd() % p_ + 1;
                    }
                    const double diff = std::abs(
                        grad_arr[j] -
                        compute_gth_gradient_item(data_arr, params_arr, resp_arr, pred_ptr, j));
                    REQUIRE(diff < tol);
                }
            }
            else {
                for (std::int64_t j = 0; j <= p_; ++j) {
                    const double diff = std::abs(
                        grad_arr[j] -
                        compute_gth_gradient_item(data_arr, params_arr, resp_arr, pred_ptr, j));
                    REQUIRE(diff < tol);
                }
            }
        }
    }

    Float compute_gth_hessian_item(array<Float>& data_arr,
                                   array<Float>& params_arr,
                                   array<std::int32_t>& resp_arr,
                                   const Float* pred_ptr,
                                   std::int64_t j,
                                   std::int64_t k) {
        if (!fit_intercept_ && (j == 0 || k == 0)) {
            return 0.0;
        }
        Float ans = 0;
        for (std::int64_t i = 0; i < n_; ++i) {
            Float x1 = j == 0 ? 1 : data_arr[i * p_ + j - 1];
            Float x2 = k == 0 ? 1 : data_arr[i * p_ + k - 1];
            ans += x1 * x2 * pred_ptr[i] * (1 - pred_ptr[i]);
        }
        ans /= n_;
        // We do not apply regularization to w_0
        ans += (k == j && j > 0) ? L2_ * 2 : 0;
        return ans;
    }

    void check_hessian(const objective_function::compute_result<>& result,
                       array<Float>& data_arr,
                       array<Float>& params_arr,
                       array<std::int32_t>& resp_arr,
                       const Float* pred_ptr,
                       const double tol = 1e-4,
                       const std::int32_t stochastic = 0) {
        std::mt19937 rnd(2007 + n_ + p_ + n_ * p_ + 2);
        if (result.get_result_options().test(result_options::hessian)) {
            const auto hessian = result.get_hessian();
            REQUIRE(hessian.get_row_count() == p_ + 1);
            REQUIRE(hessian.get_column_count() == p_ + 1);
            auto hess_arr = row_accessor<const Float>(hessian).pull({ 0, -1 });
            if (stochastic > 0) {
                for (std::int32_t num_checks = 0; num_checks < stochastic; ++num_checks) {
                    std::int64_t j = rnd() % (p_ + 1);
                    std::int64_t k = rnd() % (p_ + 1);
                    if (!fit_intercept_) {
                        j = rnd() % p_ + 1;
                        k = rnd() % p_ + 1;
                    }
                    const double diff = std::abs(
                        hess_arr[j * (p_ + 1) + k] -
                        compute_gth_hessian_item(data_arr, params_arr, resp_arr, pred_ptr, j, k));
                    REQUIRE(diff < tol);
                }
            }
            else {
                for (std::int64_t j = 0; j <= p_; ++j) {
                    for (std::int64_t k = 0; k <= p_; ++k) {
                        const double diff = std::abs(hess_arr[j * (p_ + 1) + k] -
                                                     compute_gth_hessian_item(data_arr,
                                                                              params_arr,
                                                                              resp_arr,
                                                                              pred_ptr,
                                                                              j,
                                                                              k));
                        REQUIRE(diff < tol);
                    }
                }
            }
        }
    }

    void check_compute_result(const objective_function::compute_result<>& result) {
        array<Float> data_arr = row_accessor<const Float>(data_).pull({ 0, -1 });
        array<Float> params_arr = row_accessor<const Float>(params_).pull({ 0, -1 });
        array<std::int32_t> resp_arr = row_accessor<const std::int32_t>(responses_).pull({ 0, -1 });

        auto pred_arr = array<Float>::zeros(n_);
        auto* const pred_ptr = pred_arr.get_mutable_data();

        calculate_predictions(data_arr, params_arr, pred_ptr);

        const double tol = te::get_tolerance<Float>(1e-4, 1e-6);

        check_value(result, params_arr, resp_arr, pred_ptr, tol);

        check_gradient(result, data_arr, params_arr, resp_arr, pred_ptr, tol);

        check_hessian(result, data_arr, params_arr, resp_arr, pred_ptr, tol);
    }

    void stochastic_checks(const objective_function::compute_result<>& result,
                           std::int32_t gradient_checks = 5,
                           std::int32_t hessian_checks = 25) {
        auto data_arr = row_accessor<const Float>(data_).pull({ 0, -1 });
        auto params_arr = row_accessor<const Float>(params_).pull({ 0, -1 });
        auto resp_arr = row_accessor<const std::int32_t>(responses_).pull({ 0, -1 });

        auto pred_arr = array<Float>::zeros(n_);
        auto* const pred_ptr = pred_arr.get_mutable_data();

        calculate_predictions(data_arr, params_arr, pred_ptr);

        const double tol = te::get_tolerance<Float>(1e-2, 1e-6);

        check_value(result, params_arr, resp_arr, pred_ptr, tol);

        check_gradient(result, data_arr, params_arr, resp_arr, pred_ptr, tol, gradient_checks);

        check_hessian(result, data_arr, params_arr, resp_arr, pred_ptr, tol, hessian_checks);
    }

protected:
    std::int64_t n_ = 20;
    std::int64_t p_ = 10;
    table data_;
    table params_;
    table responses_;
    Float L1_ = 0;
    Float L2_ = 0;
    bool fit_intercept_ = true;
};

using logloss_types = COMBINE_TYPES((float, double), (obj_fun::method::dense_batch));

} // namespace oneapi::dal::objective_function::test
