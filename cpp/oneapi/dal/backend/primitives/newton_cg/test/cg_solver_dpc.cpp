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
#include "oneapi/dal/backend/primitives/newton_cg/cg_solver.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/backend/primitives/rng/rng_engine.hpp"


#include "oneapi/dal/backend/primitives/rng/rng_engine.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <typename Param>
class cg_solver_test : public te::float_algo_fixture<Param> {
public:
    using float_t = Param;

    void gen_input(std::int64_t n) {

        n_ = n;
        
        auto A_host =
            ndarray<float_t, 2>::empty(this->get_queue(), { n_, n_ }, sycl::usm::alloc::host);
        auto x_host =
            ndarray<float_t, 1>::empty(this->get_queue(), { n_ }, sycl::usm::alloc::host);
        auto b_host =
            ndarray<float_t, 1>::empty(this->get_queue(), { n_ }, sycl::usm::alloc::host);

        primitives::rng<float_t> rn_gen;
        primitives::engine eng(2007);

        rn_gen.uniform(n_ * n_, A_host.get_mutable_data(), eng.get_state(), -1.0, 1.0);
        rn_gen.uniform(n_, x_host.get_mutable_data(), eng.get_state(), -1.0, 1.0);

        for (std::int64_t i = 0; i < n_; ++i) {
            b_host.at(i) = 0;
            for (std::int64_t j = 0; j < n_; ++j) {
                b_host.at(i) += A_host.at(i, j) * x_host.at(j);
            }
        }

        std::cout << "Matrix A" << std::endl;
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < n_; ++j) {
                std::cout << A_host.at(i, j) << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "Vector x" << std::endl;
        for (int i = 0; i < n_; ++i) {
            std::cout << x_host.at(i) << " ";
        }
        std::cout << std::endl;


        std::cout << "Vector b" << std::endl;
        for (int i = 0; i < n_; ++i) {
            std::cout << b_host.at(i) << " ";
        }
        std::cout << std::endl;

        A_ = A_host.to_device(this->get_queue());
        x_ = x_host.to_device(this->get_queue());
        b_ = b_host.to_device(this->get_queue());
    }

    void test_cg_solver() {
        matrix_operator<float_t> mul_operator(this->get_queue(), A_);
        auto [x0, x0_init_event] = ndarray<float_t, 1>::zeros(this->get_queue(), {n_}, sycl::usm::alloc::device);
        x0_init_event.wait_and_throw();
        auto buffer = ndarray<float_t, 1>::empty(this->get_queue(), {3 * n_}, sycl::usm::alloc::device);
        auto buffer1 = buffer.get_slice(0, n_);
        auto buffer2 = buffer.get_slice(n_, 2 * n_);
        auto buffer3 = buffer.get_slice(2 * n_, 3 * n_);

        cg_solve<float_t, matrix_operator<float_t>>(this->get_queue(), 
                 mul_operator, 
                 b_, 
                 x0, 
                 buffer1, 
                 buffer2, 
                 buffer3, 1e-3, 1e-3, 10, {}).wait_and_throw();
    }

private:
    
    std::int64_t n_;

    ndarray<float_t, 2> A_;
    ndarray<float_t, 1> x_;
    ndarray<float_t, 1> b_;
};


TEMPLATE_TEST_M(cg_solver_test, "gold input test - double", "[cg-solver][gpu]", double) {
    SKIP_IF(this->not_float64_friendly());
    SKIP_IF(this->get_policy().is_cpu());
    this->gen_input(5);
    this->test_cg_solver();
    // this->test_gold_input();
}

} // namespace oneapi::dal::backend::primitives::test
