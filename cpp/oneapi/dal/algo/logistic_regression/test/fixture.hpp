/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "oneapi/dal/algo/logistic_regression/common.hpp"
#include "oneapi/dal/algo/logistic_regression/train.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
// #include "oneapi/dal/algo/logistic_regression/infer.hpp"

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

#include "oneapi/dal/test/engine/metrics/regression.hpp"
#include "oneapi/dal/backend/primitives/rng/rng_engine.hpp"

namespace oneapi::dal::logistic_regression::test {

namespace te = dal::test::engine;
namespace de = dal::detail;
namespace la = te::linalg;

namespace pr = oneapi::dal::backend::primitives;

template <typename TestType, typename Derived>
class log_reg_test : public te::crtp_algo_fixture<TestType, Derived> {
public:
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using task_t = std::tuple_element_t<2, TestType>;

    using train_input_t = train_input<task_t>;
    using train_result_t = train_result<task_t>;
    //using test_input_t = infer_input<task_t>;
    //using test_result_t = infer_result<task_t>;

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<float_t>();
    }

    Derived* get_impl() {
        return static_cast<Derived*>(this);
    }

    auto get_descriptor() const {
        result_option_id resopts = result_options::coefficients;
        if (this->fit_intercept_)
            resopts = resopts | result_options::intercept;
        return logistic_regression::descriptor<float_t, method_t, task_t>(fit_intercept_, L2_)
            .set_result_options(resopts);
    }

    void gen_dimensions(std::int64_t n = -1, std::int64_t p = -1) {
        if (n == -1 || p == -1) {
            this->n_ = 1000; // GENERATE(100, 200, 1000, 10000);
            this->p_ = 10; //GENERATE(10, 20, 30);
        }
        else {
            this->n_ = n;
            this->p_ = p;
        }
    }

    void gen_input(std::int64_t seed = 2007) {
        this->get_impl()->gen_dimensions();

        std::int64_t dim = fit_intercept_ ? p_ + 1 : p_;
        std::int64_t st_ind = fit_intercept_;

        X_host_ =
            pr::ndarray<float_t, 2>::empty(this->get_queue(), { n_, p_ }, sycl::usm::alloc::host);
        //auto y_prob =
        //    ndarray<float_t, 1>::empty(this->get_queue(), { n_ + 1 }, sycl::usm::alloc::host);
        y_host_ = pr::ndarray<std::int32_t, 1>::empty(this->get_queue(),
                                                      { n_ + 1 },
                                                      sycl::usm::alloc::host);
        params_host_ =
            pr::ndarray<float_t, 1>::empty(this->get_queue(), { dim }, sycl::usm::alloc::host);
        pr::rng<float_t> rn_gen;
        pr::engine eng(2007 + n_);
        rn_gen.uniform(n_ * p_, X_host_.get_mutable_data(), eng.get_state(), -10.0, 10.0);
        rn_gen.uniform(p_ + 1, params_host_.get_mutable_data(), eng.get_state(), -5.0, 5.0);
        for (std::int64_t i = 0; i < n_; ++i) {
            float_t val = 0;
            for (std::int64_t j = 0; j < p_; ++j) {
                val += X_host_.at(i, j) * params_host_.at(j + st_ind);
            }
            if (fit_intercept_) {
                val += params_host_.at(0);
            }
            val = float_t(1) / (1 + std::exp(-val));
            // y_prob.at(i) = val;
            if (val < 0.5) {
                y_host_.at(i) = 0;
            }
            else {
                y_host_.at(i) = 1;
            }
        }
    }

    void run_test() {
        auto y_gpu = y_host_.to_device(this->get_queue());
        auto X_gpu = X_host_.to_device(this->get_queue());

        table data = homogen_table::wrap<float_t>(X_gpu.get_mutable_data(), n_, p_);
        table labels = homogen_table::wrap<std::int32_t>(y_gpu.get_mutable_data(), n_, 1);

        const auto desc = this->get_descriptor();
        const auto train_res = this->train(desc, data, labels);
        table intercept;
        pr::ndarray<float_t, 1> bias_host;
        if (fit_intercept_) {
            intercept = train_res.get_intercept();
            bias_host = pr::table2ndarray_1d<float_t>(this->get_queue(),
                                                      intercept,
                                                      sycl::usm::alloc::device)
                            .to_host(this->get_queue());
            std::cout << bias_host.at(0) << " ";
        }
        table coefs = train_res.get_coefficients();
        auto coefs_host =
            pr::table2ndarray_1d<float_t>(this->get_queue(), coefs, sycl::usm::alloc::device)
                .to_host(this->get_queue());

        for (int i = 0; i < p_; ++i) {
            std::cout << coefs_host.at(i) << " ";
        }
        std::cout << std::endl;
    }

protected:
    bool fit_intercept_ = true;
    double L2_ = 0.0;
    std::int64_t n_;
    std::int64_t p_;
    pr::ndarray<float_t, 2> X_host_;
    pr::ndarray<float_t, 1> params_host_;
    pr::ndarray<std::int32_t, 1> y_host_;
    pr::ndarray<std::int32_t, 1> resp_;
    // table x_test_;
};

using lr_types = COMBINE_TYPES((float, double),
                               (logistic_regression::method::newton_cg),
                               (logistic_regression::task::classification));

} // namespace oneapi::dal::logistic_regression::test
