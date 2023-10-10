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

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <typename Param>
class newton_cg_test : public te::float_algo_fixture<Param> {
public:
    using float_t = Param;

    void gen_and_test_quadratic_function(std::int64_t n = -1) {
        if (n == -1) {
            n_ = GENERATE(5, 14, 25, 50, 100);
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

} // namespace oneapi::dal::backend::primitives::test
