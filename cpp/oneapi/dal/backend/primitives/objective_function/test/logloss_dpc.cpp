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

#include <type_traits>

#include "oneapi/dal/backend/primitives/objective_function/logloss.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/backend/primitives/debug.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <ndorder order>
struct order_tag {
    static constexpr ndorder value = order;
};

using c_order = order_tag<ndorder::c>;
using f_order = order_tag<ndorder::f>;

template <typename Param>
class logloss_test : public te::float_algo_fixture<Param> {
public:
    using float_t = Param;

    void check_val(const float_t real,
                   const float_t expected,
                   const float_t rtol,
                   const float_t atol) {
        REQUIRE(abs(real - expected) < atol);
        REQUIRE(abs(real - expected) / std::max(std::abs(expected), (float_t)1.0) < rtol);
    }

    void generate_input(std::int64_t n = -1, std::int64_t p = -1) {
        if (n == -1 || p == -1) {
            this->n_ = GENERATE(7, 827, 13, 216);
            this->p_ = GENERATE(4, 17, 41, 256);
        }
        else {
            this->n_ = n;
            this->p_ = p;
        }

        const auto dataframe =
            GENERATE_DATAFRAME(te::dataframe_builder{ n_, p_ }.fill_uniform(-0.5, 0.5));
        const auto parameters =
            GENERATE_DATAFRAME(te::dataframe_builder{ 1, p_ + 1 }.fill_uniform(-1, 1));
        this->data_ = dataframe.get_table(this->get_homogen_table_id());
        this->params_ = parameters.get_table(this->get_homogen_table_id());
        this->labels_ =
            ndarray<std::int32_t, 1>::empty(this->get_queue(), { n_ }, sycl::usm::alloc::host);

        std::srand(2007 + n_);
        auto* const ptr_lab = this->labels_.get_mutable_data();
        for (std::int64_t i = 0; i < n_; ++i) {
            ptr_lab[i] = std::rand() % 2;
        }
    }

    void run_test() {
        constexpr float_t L1 = 1.2;
        constexpr float_t L2 = 0.7;
        auto data_array = row_accessor<const float_t>{ this->data_ }.pull(this->get_queue());
        auto data_host = ndarray<float_t, 2>::wrap(data_array.get_data(), { n_, p_ });

        auto param_array = row_accessor<const float_t>{ this->params_ }.pull(this->get_queue());
        auto params_host = ndarray<float_t, 1>::wrap(param_array.get_data(), { p_ + 1 });

        test_input(data_host, params_host, this->labels_, L1, L2);

        SUCCEED();
    }

    void test_gold_input() {
        constexpr std::int64_t n = 5;
        constexpr std::int64_t p = 3;
        constexpr float_t data[n * p] = { 0.83731708,  -0.70899924, -1.23362082, 0.23468538,
                                          -0.10549413, 1.12902673,  -0.61035703, -1.55617932,
                                          0.60419908,  0.30589827,  0.63919892,  -0.23380754,
                                          2.38196927,  1.64158111,  0.13677077 };
        constexpr std::int32_t labels[n] = { 0, 1, 1, 0, 1 };
        constexpr float_t L1 = 2.3456;
        constexpr float_t L2 = 3.123;
        constexpr float_t cur_param[p + 1] = { -0.2, 0.1, -1, 0.4 };

        auto data_host = ndarray<float_t, 2>::wrap(data, { n, p });
        auto labels_host = ndarray<std::int32_t, 1>::wrap(labels, n);
        auto params_host = ndarray<float_t, 1>::wrap(cur_param, p + 1);

        test_input(data_host, params_host, labels_host, L1, L2);

        SUCCEED();
    }

    void test_input(const ndarray<float_t, 2>& data_host,
                    const ndarray<float_t, 1>& params_host,
                    const ndarray<std::int32_t, 1>& labels_host,
                    const float_t L1,
                    const float_t L2) {
        constexpr float_t rtol = sizeof(float_t) > 4 ? 1e-6 : 1e-4;
        constexpr float_t atol = sizeof(float_t) > 4 ? 1e-6 : 1;
        constexpr float_t atol2 = sizeof(float_t) > 4 ? 1e-6 : 1e-4;
        const std::int64_t n = data_host.get_dimension(0);
        const std::int64_t p = data_host.get_dimension(1);

        auto data_gpu = data_host.to_device(this->get_queue());
        auto labels_gpu = labels_host.to_device(this->get_queue());
        auto params_gpu = params_host.to_device(this->get_queue());

        auto out_predictions =
            ndarray<float_t, 1>::empty(this->get_queue(), { n }, sycl::usm::alloc::device);

        auto p_event =
            compute_probabilities(this->get_queue(), params_gpu, data_gpu, out_predictions, {});
        p_event.wait_and_throw();

        auto predictions_host = out_predictions.to_host(this->get_queue(), {});

        const float_t logloss = test_predictions_and_logloss(data_host,
                                                             params_host,
                                                             labels_host,
                                                             predictions_host,
                                                             L1,
                                                             L2,
                                                             rtol,
                                                             atol);

        auto [out_logloss, out_e] =
            ndarray<float_t, 1>::zeros(this->get_queue(), { 1 }, sycl::usm::alloc::device);
        sycl::event logloss_event = compute_logloss(this->get_queue(),
                                                    params_gpu,
                                                    data_gpu,
                                                    labels_gpu,
                                                    out_logloss,
                                                    L1,
                                                    L2,
                                                    { out_e });
        logloss_event.wait_and_throw();
        const float_t val_logloss1 = out_logloss.to_host(this->get_queue(), {}).at(0);

        check_val(val_logloss1, logloss, rtol, atol);

        auto fill_event = fill<float_t>(this->get_queue(), out_logloss, float_t(0), {});
        auto [out_derivative, out_der_e] =
            ndarray<float_t, 1>::zeros(this->get_queue(), { p + 1 }, sycl::usm::alloc::device);
        auto logloss_event_der = compute_logloss_with_der(this->get_queue(),
                                                          params_gpu,
                                                          data_gpu,
                                                          labels_gpu,
                                                          out_predictions,
                                                          out_logloss,
                                                          out_derivative,
                                                          L1,
                                                          L2,
                                                          { fill_event, out_der_e });
        logloss_event_der.wait_and_throw();
        auto out_derivative_host = out_derivative.to_host(this->get_queue());
        const float_t val_logloss2 = out_logloss.to_host(this->get_queue(), {}).at(0);
        check_val(val_logloss2, logloss, rtol, atol);

        auto [out_derivative2, out_der_e2] =
            ndarray<float_t, 1>::zeros(this->get_queue(), { p + 1 }, sycl::usm::alloc::device);
        auto der_event = compute_derivative(this->get_queue(),
                                            params_gpu,
                                            data_gpu,
                                            labels_gpu,
                                            out_predictions,
                                            out_derivative2,
                                            L1,
                                            L2,
                                            { out_der_e2 });
        der_event.wait_and_throw();
        auto out_derivative_host2 = out_derivative2.to_host(this->get_queue());

        for (auto i = 0; i <= p; ++i) {
            REQUIRE(abs(out_derivative_host.at(i) - out_derivative_host2.at(i)) < atol);
        }

        auto [out_hessian, out_hess_e] = ndarray<float_t, 2>::zeros(this->get_queue(),
                                                                    { p + 1, p + 1 },
                                                                    sycl::usm::alloc::device);
        auto hess_event = compute_hessian(this->get_queue(),
                                          params_gpu,
                                          data_gpu,
                                          labels_gpu,
                                          out_predictions,
                                          out_hessian,
                                          L1,
                                          L2,
                                          { out_hess_e });

        auto hessian_host = out_hessian.to_host(this->get_queue(), { hess_event });
        test_formula_derivative(data_host,
                                predictions_host,
                                params_host,
                                labels_host,
                                out_derivative_host,
                                L1,
                                L2,
                                rtol,
                                atol2);
        test_formula_hessian(data_host, predictions_host, hessian_host, L2, rtol, atol2);
        test_derivative_and_hessian(data_gpu,
                                    labels_gpu,
                                    out_derivative_host,
                                    hessian_host,
                                    params_host,
                                    L1,
                                    L2,
                                    rtol,
                                    atol);
    }

    float_t test_predictions_and_logloss(const ndarray<float_t, 2>& data_host,
                                         const ndarray<float_t, 1>& params_host,
                                         const ndarray<std::int32_t, 1>& labels_host,
                                         const ndarray<float_t, 1>& probabilities,
                                         const float_t L1,
                                         const float_t L2,
                                         const float_t rtol = 1e-3,
                                         const float_t atol = 1e-3) {
        const std::int64_t n = data_host.get_dimension(0);
        const std::int64_t p = data_host.get_dimension(1);

        float_t logloss = 0;
        for (std::int64_t i = 0; i < n; ++i) {
            float_t pred = 0;
            for (std::int64_t j = 0; j < p; ++j) {
                pred += params_host.at(j + 1) * data_host.at(i, j);
            }
            pred += params_host.at(0);
            float_t prob = 1 / (1 + std::exp(-pred));
            logloss -=
                labels_host.at(i) * std::log(prob) + (1 - labels_host.at(i)) * std::log(1 - prob);
            float_t out_val = probabilities.at(i);
            REQUIRE(abs(out_val - prob) < atol);
        }
        for (std::int64_t i = 0; i <= p; ++i) {
            logloss += L1 * abs(params_host.at(i));
            logloss += L2 * params_host.at(i) * params_host.at(i);
        }
        return logloss;
    }

    double naive_logloss(const ndarray<float_t, 2>& data_host,
                         const ndarray<float_t, 1>& params_host,
                         const ndarray<std::int32_t, 1>& labels_host,
                         const float_t L1,
                         const float_t L2) {
        const std::int64_t n = data_host.get_dimension(0);
        const std::int64_t p = data_host.get_dimension(1);

        double logloss = 0;
        for (std::int64_t i = 0; i < n; ++i) {
            double pred = 0;
            for (std::int64_t j = 0; j < p; ++j) {
                pred += (double)params_host.at(j + 1) * (double)data_host.at(i, j);
            }
            pred += (double)params_host.at(0);
            logloss += std::log(1 + std::exp(-(2 * labels_host.at(i) - 1) * pred));
        }
        for (std::int64_t i = 0; i <= p; ++i) {
            logloss += L1 * abs(params_host.at(i));
            logloss += L2 * params_host.at(i) * params_host.at(i);
        }
        return logloss;
    }

    void naive_derivative(const ndarray<float_t, 2>& data,
                          const ndarray<float_t, 1>& probabilities,
                          const ndarray<float_t, 1>& params,
                          const ndarray<std::int32_t, 1>& labels,
                          ndarray<double, 1>& out_der,
                          float_t L1,
                          float_t L2) {
        const std::int64_t n = data.get_dimension(0);
        const std::int64_t p = data.get_dimension(1);
        for (std::int64_t j = 0; j <= p; ++j) {
            double val = 0;
            for (std::int64_t i = 0; i < n; ++i) {
                double x1 = j > 0 ? data.at(i, j - 1) : 1;
                double prob = probabilities.at(i);
                val += (prob - labels.at(i)) * x1;
            }
            double param = params.at(j);
            val += L2 * 2 * param + std::copysign(L1, param);
            out_der.at(j) = val;
        }
    }

    void naive_hessian(const ndarray<float_t, 2>& data_host,
                       const ndarray<float_t, 1>& probabilities_host,
                       ndarray<double, 2>& out_hessian,
                       float_t L2) {
        const std::int64_t n = data_host.get_dimension(0);
        const std::int64_t p = data_host.get_dimension(1);
        for (std::int64_t j = 0; j <= p; ++j) {
            for (std::int64_t k = 0; k <= p; ++k) {
                double val = 0;
                for (std::int64_t i = 0; i < n; ++i) {
                    double x1 = j > 0 ? data_host.at(i, j - 1) : 1;
                    double x2 = k > 0 ? data_host.at(i, k - 1) : 1;
                    double prob = probabilities_host.at(i);
                    val += x1 * x2 * (1 - prob) * prob;
                }
                out_hessian.at(j, k) = val;
            }
            out_hessian.at(j, j) += 2 * L2;
        }
    }

    void test_formula_derivative(const ndarray<float_t, 2>& data,
                                 const ndarray<float_t, 1>& probabilities,
                                 const ndarray<float_t, 1>& params,
                                 const ndarray<std::int32_t, 1>& labels,
                                 const ndarray<float_t, 1>& derivative,
                                 const float_t L1,
                                 const float_t L2,
                                 const float_t rtol = 1e-3,
                                 const float_t atol = 1e-3) {
        const std::int64_t p = data.get_dimension(1);
        auto out_derivative =
            ndarray<double, 1>::empty(this->get_queue(), { p + 1 }, sycl::usm::alloc::host);

        naive_derivative(data, probabilities, params, labels, out_derivative, L1, L2);

        for (std::int64_t i = 0; i < p + 1; ++i) {
            check_val(out_derivative.at(i), derivative.at(i), rtol, atol);
        }
    }

    void test_formula_hessian(const ndarray<float_t, 2>& data,
                              const ndarray<float_t, 1>& probabilities,
                              const ndarray<float_t, 2>& hessian,
                              const float_t L2,
                              const float_t rtol = 1e-3,
                              const float_t atol = 1e-3) {
        const std::int64_t p = data.get_dimension(1);
        auto out_hessian =
            ndarray<double, 2>::empty(this->get_queue(), { p + 1, p + 1 }, sycl::usm::alloc::host);

        naive_hessian(data, probabilities, out_hessian, L2);

        for (std::int64_t i = 0; i <= p; ++i) {
            for (std::int64_t j = 0; j <= p; ++j) {
                check_val(out_hessian.at(i, j), hessian.at(i, j), rtol, atol);
            }
        }
    }

    void test_derivative_and_hessian(const ndarray<float_t, 2>& data,
                                     const ndarray<std::int32_t, 1>& labels,
                                     const ndarray<float_t, 1>& derivative,
                                     const ndarray<float_t, 2>& hessian,
                                     const ndarray<float_t, 1>& params_host,
                                     const float_t L1,
                                     const float_t L2,
                                     const float_t rtol = 1e-3,
                                     const float_t atol = 1e-3) {
        const std::int64_t n = data.get_dimension(0);
        const std::int64_t p = data.get_dimension(1);
        constexpr std::int64_t max_n = 2000;
        constexpr float_t step = sizeof(float_t) > 4 ? 1e-4 : 1e-3;

        const auto data_host = data.to_host(this->get_queue());
        const auto labels_host = labels.to_host(this->get_queue());

        std::array<float_t, max_n> cur_param;
        for (std::int64_t i = 0; i < p + 1; ++i) {
            cur_param[i] = params_host.at(i);
        }

        auto out_logloss =
            ndarray<float_t, 1>::empty(this->get_queue(), { 1 }, sycl::usm::alloc::device);
        auto out_predictions =
            ndarray<float_t, 1>::empty(this->get_queue(), { n }, sycl::usm::alloc::device);
        auto out_derivative_up =
            ndarray<float_t, 1>::empty(this->get_queue(), { p + 1 }, sycl::usm::alloc::device);
        auto out_derivative_down =
            ndarray<float_t, 1>::empty(this->get_queue(), { p + 1 }, sycl::usm::alloc::device);

        for (std::int64_t i = 0; i < p + 1; ++i) {
            auto fill_event_1 = fill<float_t>(this->get_queue(), out_logloss, float_t(0), {});
            auto fill_event_2 = fill<float_t>(this->get_queue(), out_derivative_up, float_t(0), {});
            auto fill_event_3 =
                fill<float_t>(this->get_queue(), out_derivative_down, float_t(0), {});

            cur_param[i] = params_host.at(i) + step;
            auto params_host_up = ndarray<float_t, 1>::wrap(cur_param.begin(), p + 1);
            auto params_gpu_up = params_host_up.to_device(this->get_queue());

            // Compute logloss and derivative with params [w0, w1, ... w_i + eps, ...., w_p]

            sycl::event pred_up_event =
                compute_probabilities(this->get_queue(), params_gpu_up, data, out_predictions, {});
            sycl::event der_event_up =
                compute_logloss_with_der(this->get_queue(),
                                         params_gpu_up,
                                         data,
                                         labels,
                                         out_predictions,
                                         out_logloss,
                                         out_derivative_up,
                                         L1,
                                         L2,
                                         { fill_event_1, fill_event_2, pred_up_event });
            der_event_up.wait_and_throw();
            double logloss_up = naive_logloss(data_host, params_host_up, labels_host, L1, L2);
            auto der_up_host = out_derivative_up.to_host(this->get_queue(), {});

            cur_param[i] = params_host.at(i) - step;

            auto params_host_down = ndarray<float_t, 1>::wrap(cur_param.begin(), p + 1);
            auto params_gpu_down = params_host_down.to_device(this->get_queue());
            auto fill_event_4 = fill<float_t>(this->get_queue(), out_logloss, float_t(0), {});

            // Compute logloss and derivative with params [w0, w1, ... w_i - eps, ...., w_p]

            sycl::event pred_down_event = compute_probabilities(this->get_queue(),
                                                                params_gpu_down,
                                                                data,
                                                                out_predictions,
                                                                {});
            sycl::event der_event_down =
                compute_logloss_with_der(this->get_queue(),
                                         params_gpu_down,
                                         data,
                                         labels,
                                         out_predictions,
                                         out_logloss,
                                         out_derivative_down,
                                         L1,
                                         L2,
                                         { fill_event_3, fill_event_4, pred_down_event });
            der_event_down.wait_and_throw();

            double logloss_down = naive_logloss(data_host, params_host_down, labels_host, L1, L2);
            auto der_down_host = out_derivative_down.to_host(this->get_queue(), {});
            // Check condition: (logloss(w_i + eps) - logloss(w_i - eps)) / 2eps ~ d logloss / dw_i
            check_val(derivative.at(i), (logloss_up - logloss_down) / (2 * step), rtol, atol);

            if (sizeof(float_t) > 4) {
                for (std::int64_t j = 0; j < p + 1; ++j) {
                    // Check condition (d logloss(w_i + eps) / d w_j - d logloss(w_i - eps) / d w_j) / 2eps ~ h_i,j
                    // due to lack of precision this condition is not checked for 32-bit floating point numbers
                    check_val(hessian.at(i, j),
                              (der_up_host.at(j) - der_down_host.at(j)) / (2 * step),
                              rtol,
                              atol);
                }
            }
            cur_param[i] += step;
        }
    }

private:
    std::int64_t n_;
    std::int64_t p_;
    table data_;
    table params_;
    ndarray<std::int32_t, 1> labels_;
};

TEMPLATE_TEST_M(logloss_test, "gold input test - double", "[logloss]", double) {
    this->test_gold_input();
}
TEMPLATE_TEST_M(logloss_test, "gold input test - float", "[logloss]", float) {
    this->test_gold_input();
}

TEMPLATE_TEST_M(logloss_test, "test random input - double", "[logloss]", double) {
    this->generate_input();
    this->run_test();
}

TEMPLATE_TEST_M(logloss_test, "test random input - float", "[logloss]", float) {
    this->generate_input();
    this->run_test();
}

} // namespace oneapi::dal::backend::primitives::test
