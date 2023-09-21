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
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/detail/debug.hpp"

#include "oneapi/dal/backend/primitives/rng/rng_engine.hpp"

namespace oneapi::dal::backend::primitives::test {

using oneapi::dal::detail::operator<<;

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

    void run_test(const float_t L1 = 0,
                  const float_t L2 = 0,
                  bool fit_intercept = true,
                  bool batch_test = false) {
        auto data_array = row_accessor<const float_t>{ this->data_ }.pull(this->get_queue());
        auto data_host = ndarray<float_t, 2>::wrap(data_array.get_data(), { n_, p_ });

        std::int64_t dim = fit_intercept ? this->p_ + 1 : this->p_;

        auto param_array = row_accessor<const float_t>{ this->params_ }.pull(this->get_queue());
        auto params_host = ndarray<float_t, 1>::wrap(param_array.get_data(), { dim });
        test_input(data_host, params_host, this->labels_, L1, L2, fit_intercept, batch_test);

        SUCCEED();
    }

    void test_gold_input(bool fit_intercept = true) {
        constexpr std::int64_t n = 5;
        constexpr std::int64_t p = 3;
        constexpr float_t data[n * p] = { 0.83731708,  -0.70899924, -1.23362082, 0.23468538,
                                          -0.10549413, 1.12902673,  -0.61035703, -1.55617932,
                                          0.60419908,  0.30589827,  0.63919892,  -0.23380754,
                                          2.38196927,  1.64158111,  0.13677077 };
        constexpr std::int32_t labels[n] = { 0, 1, 1, 0, 1 };
        constexpr float_t L1 = 0;
        constexpr float_t L2 = 3.123;
        constexpr float_t cur_param[p + 1] = { -0.2, 0.1, -1, 0.4 };

        auto data_host = ndarray<float_t, 2>::wrap(data, { n, p });

        this->data_ = homogen_table::wrap(data_host.get_data(), n, p);

        auto labels_host = ndarray<std::int32_t, 1>::wrap(labels, n);

        ndarray<float_t, 1> params_host;
        if (fit_intercept) {
            params_host = ndarray<float_t, 1>::wrap(cur_param, p + 1);
        }
        else {
            params_host = ndarray<float_t, 1>::wrap(cur_param + 1, p);
        }

        test_input(data_host, params_host, labels_host, L1, L2, fit_intercept);

        SUCCEED();
    }

    void test_input(const ndarray<float_t, 2>& data_host,
                    const ndarray<float_t, 1>& params_host,
                    const ndarray<std::int32_t, 1>& labels_host,
                    const float_t L1,
                    const float_t L2,
                    bool fit_intercept,
                    bool batch_test = false) {
        constexpr float_t rtol = sizeof(float_t) > 4 ? 1e-6 : 1e-4;
        constexpr float_t atol = sizeof(float_t) > 4 ? 1e-6 : 1e-1;
        const std::int64_t n = data_host.get_dimension(0);
        const std::int64_t p = data_host.get_dimension(1);
        const std::int64_t dim = params_host.get_dimension(0);

        auto data_gpu = data_host.to_device(this->get_queue());
        auto labels_gpu = labels_host.to_device(this->get_queue());
        auto params_gpu = params_host.to_device(this->get_queue());

        auto out_predictions =
            ndarray<float_t, 1>::empty(this->get_queue(), { n }, sycl::usm::alloc::device);

        auto p_event = compute_probabilities(this->get_queue(),
                                             params_gpu,
                                             data_gpu,
                                             out_predictions,
                                             fit_intercept,
                                             {});
        p_event.wait_and_throw();
        auto predictions_host = out_predictions.to_host(this->get_queue(), {});

        const float_t logloss = test_predictions_and_logloss(data_host,
                                                             params_host,
                                                             labels_host,
                                                             predictions_host,
                                                             L1,
                                                             L2,
                                                             fit_intercept,
                                                             rtol,
                                                             atol);

        auto [out_logloss, out_e] =
            ndarray<float_t, 1>::zeros(this->get_queue(), { 1 }, sycl::usm::alloc::device);
        sycl::event logloss_event = compute_logloss(this->get_queue(),
                                                    labels_gpu,
                                                    out_predictions,
                                                    out_logloss,
                                                    fit_intercept,
                                                    { out_e });
        sycl::event logloss_reg_event = add_regularization_loss(this->get_queue(),
                                                                params_gpu,
                                                                out_logloss,
                                                                L1,
                                                                L2,
                                                                fit_intercept,
                                                                { logloss_event });
        logloss_reg_event.wait_and_throw();
        const float_t val_logloss1 = out_logloss.to_host(this->get_queue(), {}).at(0);

        check_val(val_logloss1, logloss, rtol, atol);

        auto fill_event = fill<float_t>(this->get_queue(), out_logloss, float_t(0), {});
        auto [out_derivative, out_der_e] =
            ndarray<float_t, 1>::zeros(this->get_queue(), { dim }, sycl::usm::alloc::device);
        auto logloss_event_der = compute_logloss_with_der(this->get_queue(),
                                                          data_gpu,
                                                          labels_gpu,
                                                          out_predictions,
                                                          out_logloss,
                                                          out_derivative,
                                                          fit_intercept,
                                                          { fill_event, out_der_e });
        auto regul_logloss_and_der_event = add_regularization_gradient_loss(this->get_queue(),
                                                                            params_gpu,
                                                                            out_logloss,
                                                                            out_derivative,
                                                                            L1,
                                                                            L2,
                                                                            fit_intercept,
                                                                            { logloss_event_der });
        regul_logloss_and_der_event.wait_and_throw();
        auto out_derivative_host = out_derivative.to_host(this->get_queue());

        const float_t val_logloss2 = out_logloss.to_host(this->get_queue(), {}).at(0);

        check_val(val_logloss2, logloss, rtol, atol);
        auto [out_derivative2, out_der_e2] =
            ndarray<float_t, 1>::zeros(this->get_queue(), { dim }, sycl::usm::alloc::device);
        auto der_event = compute_derivative(this->get_queue(),
                                            data_gpu,
                                            labels_gpu,
                                            out_predictions,
                                            out_derivative2,
                                            fit_intercept,
                                            { out_der_e2 });
        auto der_reg_event = add_regularization_gradient(this->get_queue(),
                                                         params_gpu,
                                                         out_derivative2,
                                                         L1,
                                                         L2,
                                                         fit_intercept,
                                                         { der_event });

        der_reg_event.wait_and_throw();
        auto out_derivative_host2 = out_derivative2.to_host(this->get_queue());

        for (auto i = 0; i < dim; ++i) {
            REQUIRE(abs(out_derivative_host.at(i) - out_derivative_host2.at(i)) < atol);
        }
        auto [out_hessian, out_hess_e] = ndarray<float_t, 2>::zeros(this->get_queue(),
                                                                    { p + 1, p + 1 },
                                                                    sycl::usm::alloc::device);
        auto hess_event = compute_hessian(this->get_queue(),
                                          data_gpu,
                                          labels_gpu,
                                          out_predictions,
                                          out_hessian,
                                          L1,
                                          L2,
                                          fit_intercept,
                                          { out_hess_e });

        auto hessian_host = out_hessian.to_host(this->get_queue(), { hess_event });

        test_formula_derivative(data_host,
                                predictions_host,
                                params_host,
                                labels_host,
                                out_derivative_host,
                                L1,
                                L2,
                                fit_intercept,
                                rtol,
                                atol);

        test_formula_hessian(data_host,
                             predictions_host,
                             hessian_host,
                             L2,
                             fit_intercept,
                             rtol,
                             atol);

        if (L1 == 0) {
            std::int64_t bsz = -1;
            if (batch_test) {
                bsz = GENERATE(4, 8, 16, 20, 37, 512);
            }

            // LogLossFunction has different regularization so we need to multiply it by 2 to allign with other implementations
            auto functor = LogLossFunction<float_t>(this->get_queue(),
                                                    data_,
                                                    labels_gpu,
                                                    L2 * 2,
                                                    fit_intercept,
                                                    bsz);
            auto set_point_event = functor.update_x(params_gpu, true, {});
            wait_or_pass(set_point_event).wait_and_throw();

            check_val(logloss, functor.get_value(), rtol, atol);
            auto grad_func = functor.get_gradient();
            auto grad_func_host = grad_func.to_host(this->get_queue());

            int dim = fit_intercept ? p + 1 : p;
            for (int i = 0; i < dim; ++i) {
                check_val(out_derivative_host.at(i), grad_func_host.at(i), rtol, atol);
            }
            BaseMatrixOperator<float_t>& hessp = functor.get_hessian_product();
            test_hessian_product(hessian_host, hessp, fit_intercept, L2, rtol, atol);
        }
    }

    float_t test_predictions_and_logloss(const ndview<float_t, 2>& data_host,
                                         const ndview<float_t, 1>& params_host,
                                         const ndview<std::int32_t, 1>& labels_host,
                                         const ndview<float_t, 1>& probabilities,
                                         const float_t L1,
                                         const float_t L2,
                                         bool fit_intercept,
                                         const float_t rtol = 1e-3,
                                         const float_t atol = 1e-3) {
        const std::int64_t n = data_host.get_dimension(0);
        const std::int64_t p = data_host.get_dimension(1);
        const std::int64_t start_ind = fit_intercept ? 1 : 0;
        float_t logloss = 0;
        for (std::int64_t i = 0; i < n; ++i) {
            float_t pred = 0;

            for (std::int64_t j = 0; j < p; ++j) {
                pred += params_host.at(j + start_ind) * data_host.at(i, j);
            }
            if (fit_intercept) {
                pred += params_host.at(0);
            }
            float_t prob = 1 / (1 + std::exp(-pred));
            logloss -=
                labels_host.at(i) * std::log(prob) + (1 - labels_host.at(i)) * std::log(1 - prob);
            float_t out_val = probabilities.at(i);
            REQUIRE(abs(out_val - prob) < atol);
        }
        for (std::int64_t i = 0; i < p; ++i) {
            float_t param = params_host.at(i + start_ind);
            logloss += L1 * abs(param);
            logloss += L2 * param * param;
        }
        return logloss;
    }

    double naive_logloss(const ndview<float_t, 2>& data_host,
                         const ndview<float_t, 1>& params_host,
                         const ndview<std::int32_t, 1>& labels_host,
                         const float_t L1,
                         const float_t L2,
                         bool fit_intercept) {
        const std::int64_t n = data_host.get_dimension(0);
        const std::int64_t p = data_host.get_dimension(1);

        double logloss = 0;
        for (std::int64_t i = 0; i < n; ++i) {
            double pred = 0;
            for (std::int64_t j = 0; j < p; ++j) {
                pred += (double)params_host.at(j + 1) * (double)data_host.at(i, j);
            }
            if (fit_intercept) {
                pred += (double)params_host.at(0);
            }
            logloss += std::log(1 + std::exp(-(2 * labels_host.at(i) - 1) * pred));
        }
        for (std::int64_t i = 1; i < p + 1; ++i) {
            logloss += L1 * abs(params_host.at(i));
            logloss += L2 * params_host.at(i) * params_host.at(i);
        }
        return logloss;
    }

    void naive_derivative(const ndview<float_t, 2>& data,
                          const ndview<float_t, 1>& probabilities,
                          const ndview<float_t, 1>& params,
                          const ndview<std::int32_t, 1>& labels,
                          ndview<double, 1>& out_der,
                          float_t L1,
                          float_t L2,
                          bool fit_intercept) {
        const std::int64_t n = data.get_dimension(0);
        const std::int64_t dim = params.get_dimension(0);
        for (std::int64_t j = 0; j < dim; ++j) {
            double val = 0;
            for (std::int64_t i = 0; i < n; ++i) {
                double x1;
                if (fit_intercept) {
                    x1 = j > 0 ? data.at(i, j - 1) : 1;
                }
                else {
                    x1 = data.at(i, j);
                }
                double prob = probabilities.at(i);
                val += (prob - labels.at(i)) * x1;
            }
            val += j > 0 || !fit_intercept ? L2 * 2 * params.at(j) : 0;
            out_der.at(j) = val;
        }
    }

    void naive_hessian(const ndview<float_t, 2>& data_host,
                       const ndview<float_t, 1>& probabilities_host,
                       ndview<double, 2>& out_hessian,
                       float_t L2,
                       bool fit_intercept) {
        const std::int64_t n = data_host.get_dimension(0);
        const std::int64_t p = data_host.get_dimension(1);
        const std::int64_t start_ind = (fit_intercept ? 0 : 1);
        for (std::int64_t j = start_ind; j < p + 1; ++j) {
            for (std::int64_t k = start_ind; k < p + 1; ++k) {
                double val = 0;
                for (std::int64_t i = 0; i < n; ++i) {
                    double x1 = j > 0 ? data_host.at(i, j - 1) : 1;
                    double x2 = k > 0 ? data_host.at(i, k - 1) : 1;
                    double prob = probabilities_host.at(i);
                    val += x1 * x2 * (1 - prob) * prob;
                }
                out_hessian.at(j, k) = val;
            }
            if (j > 0) {
                out_hessian.at(j, j) += 2 * L2;
            }
        }
        if (!fit_intercept) {
            for (std::int64_t j = 0; j < p + 1; ++j) {
                out_hessian.at(0, j) = 0;
                out_hessian.at(j, 0) = 0;
            }
        }
    }

    void test_formula_derivative(const ndview<float_t, 2>& data,
                                 const ndview<float_t, 1>& probabilities,
                                 const ndview<float_t, 1>& params,
                                 const ndview<std::int32_t, 1>& labels,
                                 const ndview<float_t, 1>& derivative,
                                 const float_t L1,
                                 const float_t L2,
                                 bool fit_intercept,
                                 const float_t rtol = 1e-3,
                                 const float_t atol = 1e-3) {
        const std::int64_t dim = params.get_dimension(0);
        auto out_derivative =
            ndarray<double, 1>::empty(this->get_queue(), { dim }, sycl::usm::alloc::host);

        naive_derivative(data,
                         probabilities,
                         params,
                         labels,
                         out_derivative,
                         L1,
                         L2,
                         fit_intercept);

        for (std::int64_t i = 0; i < dim; ++i) {
            check_val(out_derivative.at(i), derivative.at(i), rtol, atol);
        }
    }

    void test_formula_hessian(const ndview<float_t, 2>& data,
                              const ndview<float_t, 1>& probabilities,
                              const ndview<float_t, 2>& hessian,
                              const float_t L2,
                              bool fit_intercept,
                              const float_t rtol = 1e-3,
                              const float_t atol = 1e-3) {
        const std::int64_t p = data.get_dimension(1);
        auto out_hessian =
            ndarray<double, 2>::empty(this->get_queue(), { p + 1, p + 1 }, sycl::usm::alloc::host);

        naive_hessian(data, probabilities, out_hessian, L2, fit_intercept);

        for (std::int64_t i = 0; i <= p; ++i) {
            for (std::int64_t j = 0; j <= p; ++j) {
                check_val(out_hessian.at(i, j), hessian.at(i, j), rtol, atol);
            }
        }
    }

    void test_hessian_product(const ndview<float_t, 2>& hessian_host,
                              BaseMatrixOperator<float_t>& hessp,
                              bool fit_intercept,
                              double L2,
                              const float_t rtol = 1e-3,
                              const float_t atol = 1e-3,
                              std::int32_t num_checks = 5) {
        const std::int64_t p = hessian_host.get_dimension(0) - 1;
        const std::int64_t dim = fit_intercept ? p + 1 : p;

        primitives::rng<float_t> rn_gen;
        auto vec_host =
            ndarray<float_t, 1>::empty(this->get_queue(), { dim }, sycl::usm::alloc::host);

        for (std::int32_t ij = 0; ij < num_checks; ++ij) {
            primitives::engine eng(2007 + dim * num_checks + ij);
            rn_gen.uniform(dim, vec_host.get_mutable_data(), eng.get_state(), -1.0, 1.0);
            auto vec_gpu = vec_host.to_device(this->get_queue());
            auto out_vector =
                ndarray<float_t, 1>::empty(this->get_queue(), { dim }, sycl::usm::alloc::device);
            hessp(vec_gpu, out_vector, {}).wait_and_throw();

            auto out_vector_host = out_vector.to_host(this->get_queue());
            const std::int64_t st = fit_intercept ? 0 : 1;

            for (std::int64_t i = st; i < p + 1; ++i) {
                float_t correct = 0;
                for (std::int64_t j = st; j < p + 1; ++j) {
                    correct += vec_host.at(j - st) * hessian_host.at(i, j);
                }
                check_val(out_vector_host.at(i - st), correct, rtol, atol);
            }
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
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->test_gold_input();
}

TEMPLATE_TEST_M(logloss_test, "gold input test - double - no fit_intercept", "[logloss]", double) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->test_gold_input(false);
}

TEMPLATE_TEST_M(logloss_test, "gold input test - float", "[logloss]", float) {
    SKIP_IF(this->get_policy().is_cpu());
    this->test_gold_input();
}

TEMPLATE_TEST_M(logloss_test, "gold input test - float - no fit intercept", "[logloss]", float) {
    SKIP_IF(this->get_policy().is_cpu());
    this->test_gold_input(false);
}

TEMPLATE_TEST_M(logloss_test, "test random input - double without L1", "[logloss]", double) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_input();
    this->run_test(0.0, 1.3);
}

TEMPLATE_TEST_M(logloss_test,
                "test random input - double without L1 - no fit intercept",
                "[logloss]",
                double) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_input();
    this->run_test(0.0, 1.3, false);
}

TEMPLATE_TEST_M(logloss_test, "batch test - double", "[logloss]", double) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_input();
    this->run_test(0.0, 1.3, true, true);
}

TEMPLATE_TEST_M(logloss_test, "batch test - double - no fit intercept", "[logloss]", double) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_input();
    this->run_test(0.0, 1.3, false, true);
}

TEMPLATE_TEST_M(logloss_test, "test random input - double with L1", "[logloss]", double) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_input();
    this->run_test(0.4, 1.3);
}

TEMPLATE_TEST_M(logloss_test,
                "test random input - double with L1 -- no fit intercept",
                "[logloss]",
                double) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_input();
    this->run_test(0.4, 1.3, false);
}

TEMPLATE_TEST_M(logloss_test, "test random input - float", "[logloss]", float) {
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_input();
    this->run_test(0.4, 1.3);
}

TEMPLATE_TEST_M(logloss_test, "test random input - float - no fit intercept", "[logloss]", float) {
    SKIP_IF(this->get_policy().is_cpu());
    this->generate_input();
    this->run_test(0.4, 1.3, false);
}

} // namespace oneapi::dal::backend::primitives::test
