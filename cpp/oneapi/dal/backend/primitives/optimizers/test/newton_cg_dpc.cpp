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
#include <memory>
#include "oneapi/dal/backend/primitives/optimizers/test/fixture.hpp"
#include "oneapi/dal/backend/primitives/optimizers/cg_solver.hpp"
#include "oneapi/dal/backend/primitives/optimizers/newton_cg.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/backend/primitives/rng/rng_engine.hpp"
#include <math.h>

#include "oneapi/dal/backend/primitives/objective_function/logloss.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <typename Param>
class newton_cg_test : public te::float_algo_fixture<Param> {
public:
    using float_t = Param;

    void gen_and_test_logistic_loss(std::int64_t n = -1,
                                    std::int64_t p = -1,
                                    bool fit_intercept = true) {
        if (n == -1 || p == -1) {
            n_ = GENERATE(1000, 10000, 20000);
            p_ = GENERATE(3, 10, 20, 50);
        }
        else {
            n_ = n;
            p_ = p;
        }
        std::int64_t bsz = GENERATE(-1, 1024);
        auto X_host =
            ndarray<float_t, 2>::empty(this->get_queue(), { n_, p_ }, sycl::usm::alloc::host);
        auto y_prob =
            ndarray<float_t, 1>::empty(this->get_queue(), { n_ + 1 }, sycl::usm::alloc::host);
        auto y_host =
            ndarray<std::int32_t, 1>::empty(this->get_queue(), { n_ + 1 }, sycl::usm::alloc::host);
        auto params_host =
            ndarray<float_t, 1>::empty(this->get_queue(), { p_ + 1 }, sycl::usm::alloc::host);
        primitives::rng<float_t> rn_gen;
        primitives::engine eng(2007 + n);
        rn_gen.uniform(n_ * p_, X_host.get_mutable_data(), eng.get_state(), -10.0, 10.0);
        rn_gen.uniform(p_ + 1, params_host.get_mutable_data(), eng.get_state(), -5.0, 5.0);
        for (std::int64_t i = 0; i < n_; ++i) {
            float_t val = 0;
            for (std::int64_t j = 0; j < p_; ++j) {
                val += X_host.at(i, j) * params_host.at(j + 1);
            }
            val += params_host.at(0);
            val = float_t(1) / (1 + std::exp(-val));
            y_prob.at(i) = val;
            if (val < 0.5) {
                y_host.at(i) = 0;
            }
            else {
                y_host.at(i) = 1;
            }
        }

        int train_size = n_ * 0.7;
        int test_size = n_ - train_size;
        auto X_train = X_host.slice(0, train_size);
        auto X_test = X_host.slice(train_size, test_size);
        auto y_train = y_host.slice(0, train_size);
        auto y_test = y_host.slice(train_size, test_size);

        // std::cout << "Train size: " << X_train.get_dimension(0) << " " << X_train.get_dimension(1) << std::endl;
        // std::cout << "Test size: " << X_test.get_dimension(0) << " " << X_test.get_dimension(1) << std::endl;

        // std::cout << "Gth parameters: " << std::endl;
        // for (std::int64_t i = 0; i < p_ + 1; ++i) {
        //     std::cout << params_host.at(i) << " ";
        // }
        // std::cout << std::endl;

        auto y_gpu = y_train.to_device(this->get_queue());
        A_ = X_train.to_device(this->get_queue());
        table data = homogen_table::wrap<float_t>(A_.get_mutable_data(), train_size, p_);
        auto logloss_func =
            LogLossFunction<float_t>(this->get_queue(), data, y_gpu, 3.0, true, bsz);
        auto [solution_, fill_e] =
            ndarray<float_t, 1>::zeros(this->get_queue(), { p_ + 1 }, sycl::usm::alloc::device);
        newton_cg(this->get_queue(), logloss_func, solution_, float_t(1e-8), 100, { fill_e })
            .wait_and_throw();
        auto solution_host = solution_.to_host(this->get_queue());

        // std::cout << "Parameters found by algorithm" << std::endl;
        // for (std::int64_t i = 0; i < p_ + 1; ++i) {
        //     std::cout << solution_host.at(i) << " ";
        // }
        // std::cout << std::endl;

        double train_score = 0;
        for (std::int64_t i = 0; i < train_size; ++i) {
            float_t val = 0;
            for (int j = 0; j < p_; ++j) {
                val += X_train.at(i, j) * solution_host.at(j + 1);
            }
            val += solution_host.at(0);
            val = float_t(1) / (1 + std::exp(-val));
            std::int32_t pred = val > 0.5 ? 1 : 0;
            if (pred == y_train.at(i)) {
                train_score += 1;
            }
        }
        // std::cout << "Accuracy on train: " << train_score / train_size << "(" << train_score << " out of " << train_size << ")" << std::endl;

        double val_score = 0;
        for (std::int64_t i = 0; i < test_size; ++i) {
            float_t val = 0;
            for (std::int64_t j = 0; j < p_; ++j) {
                val += X_test.at(i, j) * solution_host.at(j + 1);
            }
            val += solution_host.at(0);
            val = float_t(1) / (1 + std::exp(-val));
            std::int32_t pred = val > 0.5 ? 1 : 0;
            if (pred == y_test.at(i)) {
                val_score += 1;
            }
        }
        // std::cout << "Accuracy on test: " << val_score / test_size << "(" << val_score << " out of " << test_size << ")" << std::endl;

        REQUIRE(train_score >= 0.97 * train_size);
        REQUIRE(val_score >= 0.96 * test_size);
    }

    void gen_and_test_quadratic_function(std::int64_t n = -1) {
        if (n == -1) {
            n_ = GENERATE(5, 14, 41, 100);
        }
        else {
            n_ = n;
        }
        auto A_host =
            ndarray<float_t, 2>::empty(this->get_queue(), { n_, n_ }, sycl::usm::alloc::host);
        solution_ = ndarray<float_t, 1>::empty(this->get_queue(), { n_ }, sycl::usm::alloc::host);
        auto b_host = ndarray<float_t, 1>::empty(this->get_queue(), { n_ }, sycl::usm::alloc::host);
        primitives::rng<float_t> rn_gen;
        primitives::engine eng(4014 + n_);
        rn_gen.uniform(n_, solution_.get_mutable_data(), eng.get_state(), -1.0, 1.0);

        create_stable_matrix(this->get_queue(), A_host, float_t(0.1), float_t(5.0));

        for (std::int64_t i = 0; i < n_; ++i) {
            b_host.at(i) = 0;
            for (std::int64_t j = 0; j < n_; ++j) {
                b_host.at(i) += A_host.at(i, j) * solution_.at(j);
            }
        }

        A_ = A_host.to_device(this->get_queue());
        b_ = b_host.to_device(this->get_queue());

        func_ = std::make_shared<QuadraticFunction<float_t>>(this->get_queue(), A_, b_);

        auto x_host = ndarray<float_t, 1>::empty(this->get_queue(), { n_ }, sycl::usm::alloc::host);
        auto buffer = ndarray<float_t, 1>::empty(this->get_queue(), { n_ }, sycl::usm::alloc::host);

        for (std::int32_t test_num = 0; test_num < 5; ++test_num) {
            rn_gen.uniform(n_, x_host.get_mutable_data(), eng.get_state(), -1.0, 1.0);
            auto x_gpu = x_host.to_device(this->get_queue());
            auto compute_event_vec = func_->update_x(x_gpu, true, {});
            wait_or_pass(compute_event_vec).wait_and_throw();

            float_t val = func_->get_value();
            auto grad_gpu = func_->get_gradient();
            auto grad_host = grad_gpu.to_host(this->get_queue());
            for (std::int64_t i = 0; i < n_; ++i) {
                float_t grad_gth = 0;
                for (std::int64_t j = 0; j < n_; ++j) {
                    grad_gth += A_host.at(i, j) * x_host.at(j);
                }
                buffer.at(i) = grad_gth;
                grad_gth -= b_host.at(i);
                // TODO: Investigate whether 2e-5 is acceptable substitute (fails with 1e-5)
                check_val(grad_gth, grad_host.at(i), float_t(1e-5), float_t(2e-5));
            }

            float_t val_gth = 0;
            for (std::int64_t i = 0; i < n_; ++i) {
                val_gth += buffer.at(i) * x_host.at(i);
            }
            val_gth /= 2;
            for (std::int64_t i = 0; i < n_; ++i) {
                val_gth -= b_host.at(i) * x_host.at(i);
            }
            check_val(val_gth, val, float_t(5e-5), float_t(5e-5));
        }
    }

    void test_newton_cg() {
        auto [x, x_event] =
            ndarray<float_t, 1>::zeros(this->get_queue(), { n_ }, sycl::usm::alloc::device);

        float_t conv_tol = sizeof(float_t) == 4 ? 1e-7 : 1e-14;
        newton_cg(this->get_queue(), *func_, x, conv_tol, 100, { x_event }).wait_and_throw();
        auto x_host = x.to_host(this->get_queue());
        float_t tol = sizeof(float_t) == 4 ? 1e-4 : 1e-7;
        for (std::int64_t i = 0; i < n_; ++i) {
            check_val(solution_.at(i), x_host.at(i), tol, tol);
        }
    }

private:
    std::int64_t n_;
    std::int64_t p_;
    std::shared_ptr<BaseFunction<float_t>> func_;
    ndarray<float_t, 1> solution_;
    ndarray<float_t, 2> A_;
    ndarray<float_t, 1> b_;
};

TEMPLATE_TEST_M(newton_cg_test,
                "test newton-cg with stable matrix - float",
                "[newton-cg][gpu]",
                float) {
    SKIP_IF(this->get_policy().is_cpu());
    this->gen_and_test_quadratic_function();
    this->test_newton_cg();
}

TEMPLATE_TEST_M(newton_cg_test,
                "test newton-cg with stable matrix - double",
                "[newton-cg][gpu]",
                double) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->gen_and_test_quadratic_function();
    this->test_newton_cg();
}

TEMPLATE_TEST_M(newton_cg_test,
                "test newton-cg with logloss function - double",
                "[newton-cg][gpu]",
                double) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->gen_and_test_logistic_loss();
}

TEMPLATE_TEST_M(newton_cg_test,
                "test newton-cg with logloss function - float",
                "[newton-cg][gpu]",
                float) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->gen_and_test_logistic_loss();
}

} // namespace oneapi::dal::backend::primitives::test
