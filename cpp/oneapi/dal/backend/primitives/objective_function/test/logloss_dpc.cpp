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

#include <type_traits>

#include "oneapi/dal/backend/primitives/objective_function/logloss.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/backend/primitives/debug.hpp"

using std::cout;
using std::endl;

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <ndorder order>
struct order_tag {
    static constexpr ndorder value = order;
};

using c_order = order_tag<ndorder::c>;
using f_order = order_tag<ndorder::f>;

template <typename Param>
class logloss_test : public te::float_algo_fixture<std::tuple_element_t<0, Param>> {
public:
    using float_t = std::tuple_element_t<0, Param>;

    /*
    void generate() {
        n_ = GENERATE(7, 707, 1, 251, 5);
        p_ = GENERATE(17, 999, 1, 5, 1001);
        CAPTURE(n_, p_);
        generate_input();
    }

    void generate_input() {
        const auto train_dataframe =
            GENERATE_DATAFRAME(te::dataframe_builder{ n_, p_ }.fill_uniform(-0.2, 0.5));
        this->input_data_ = train_dataframe.get_table(this->get_homogen_table_id());
    }
    */

    void test_gold_input() {
        constexpr int64_t n_ = 5;
        constexpr int64_t p_ = 3;
        const float_t data[n_ * p_] = \
        {0.83731708, -0.70899924, -1.23362082, \
        0.23468538, -0.10549413, 1.12902673, \
        -0.61035703, -1.55617932,  0.60419908, \
        0.30589827, 0.63919892, -0.23380754, \
        2.38196927,  1.64158111,  0.13677077};
        const std::int32_t labels[n_] = {0, 1, 1, 0, 1};
        const float_t L1 = 2.3456;
        const float_t L2 = 3.123;
        // const Float real_param[p_ + 1] = {-0.38, 0.58, -0.4, 0.7};
        float_t cur_param[p_ + 1] = {-0.2, 0.1, -1, 0.4};
        
        const float_t real_predictions[n_] = {0.09928262 ,0.38057336, 1.53682325, -0.90213211, -1.54867588};


        auto data_host = ndarray<float_t, 2>::wrap(data, { n_, p_ });
        auto labels_host = ndarray<std::int32_t, 1>::wrap(labels, n_);
        auto params_host = ndarray<float_t, 1>::wrap(cur_param, p_ + 1);

        auto data_gpu = data_host.to_device(this->get_queue());
        auto labels_gpu = labels_host.to_device(this->get_queue());
        auto params_gpu = params_host.to_device(this->get_queue());

        auto out_predictions = ndarray<float_t, 1>::empty(this->get_queue(), { n_ },  sycl::usm::alloc::device);


        auto p_event = compute_predictions(this->get_queue(), params_gpu, data_gpu, out_predictions, {});

        p_event.wait_and_throw();


        auto predictions_host = out_predictions.to_host(this->get_queue(), {});

        cout << predictions_host;
        // cout << out_predictions;

        float_t logloss = 0;

        for (int i = 0; i < n_; ++i) {
            float_t prob = 1 / (1 + sycl::exp(-real_predictions[i]));
            logloss -= labels[i] * sycl::log(prob) + (1 - labels[i]) * sycl::log(1 - prob);
            // cout << - labels[i] * sycl::log(prob) - (1 - labels[i]) * sycl::log(1 - prob) << endl;
            // cout << sycl::log(1 + sycl::exp(- (2 * labels[i] - 1) * real_predictions[i])) << endl;;
            float_t out_val = predictions_host.at(i);
            // cout << i << ": " << out_val << endl;
            // REQUIRE(abs(out_val - real_predictions[i]) < 1e-5);
            REQUIRE(abs(out_val - prob) < 1e-5);
        }
        // cout << logloss << endl;
        for (int i = 0; i <= p_; ++i) {
            // cout << L1 * abs(cur_param[i]) + L2 * cur_param[i] * cur_param[i] << " ";
            logloss += L1 * abs(cur_param[i]);
            logloss += L2 * cur_param[i] * cur_param[i];
        }
        // cout << endl;
        // cout << logloss << endl;

        auto [out_logloss, out_e] = ndarray<float_t, 1>::zeros(this->get_queue(), {1}, sycl::usm::alloc::device);
        // auto out_logloss_.to_device(this->get_queue());

        sycl::event logloss_event = compute_logloss(this->get_queue(), params_gpu, data_gpu, labels_gpu, out_logloss, L1, L2, {out_e});

        logloss_event.wait_and_throw();
        float_t val_logloss = out_logloss.to_host(this->get_queue(), {}).at(0);
        REQUIRE(abs(val_logloss - logloss) < 1e-5);

        auto fill_event = fill<float_t>(this->get_queue(), out_logloss, float_t(0), {});
        auto [out_derivative, out_der_e] = ndarray<float_t, 1>::zeros(this->get_queue(), {p_ + 1}, sycl::usm::alloc::device);
        auto logloss_event_der = compute_logloss_with_der(this->get_queue(), params_gpu, data_gpu, labels_gpu, out_logloss, out_derivative, L1, L2, {fill_event, out_der_e});

        logloss_event_der.wait_and_throw();

        auto out_derivative_host = out_derivative.to_host(this->get_queue());
        val_logloss = out_logloss.to_host(this->get_queue(), {}).at(0);
        REQUIRE(abs(val_logloss - logloss) < 1e-5);
        cout << out_derivative_host;

        auto out_hessian = ndarray<float_t, 2>::empty(this->get_queue(), {p_ + 1, p_ + 1}, sycl::usm::alloc::device);
        auto hess_event = compute_hessian(this->get_queue(), params_gpu, data_gpu, labels_gpu, out_hessian, L1, L2, {});

        auto hessian_host = out_hessian.to_host(this->get_queue(), {hess_event});

        cout << hessian_host;

        const float_t eps = 1e-4;
        for (int i = 0; i <= p_; ++i) { // --------- CHANGE i to zero !!!! -------
            cur_param[i] += eps;
            auto params_host2 = ndarray<float_t, 1>::wrap(cur_param, p_ + 1);
            auto params_gpu2 = params_host2.to_device(this->get_queue());
            auto fill_event2 = fill<float_t>(this->get_queue(), out_logloss, float_t(0), {});

            sycl::event logloss_event2 = compute_logloss(this->get_queue(), params_gpu2, data_gpu, labels_gpu, out_logloss, L1, L2, {fill_event2});
            logloss_event2.wait_and_throw();
            float_t logloss_up = out_logloss.to_host(this->get_queue(), {}).at(0);

            auto [out_derivative2, out_der_e2] = ndarray<float_t, 1>::zeros(this->get_queue(), {p_ + 1}, sycl::usm::alloc::device);
            sycl::event der_event2 = compute_logloss_with_der(this->get_queue(), params_gpu, data_gpu, labels_gpu, out_logloss, out_derivative2, L1, L2, {out_der_e2});
            der_event2.wait_and_throw();

            auto out_der_ptr2 = out_derivative2.to_host(this->get_queue(), {});

            cur_param[i] -= 2 * eps;

            auto params_host3 = ndarray<float_t, 1>::wrap(cur_param, p_ + 1);
            auto params_gpu3 = params_host3.to_device(this->get_queue());
            auto fill_event3 = fill<float_t>(this->get_queue(), out_logloss, float_t(0), {});

            sycl::event logloss_event3 = compute_logloss(this->get_queue(), params_gpu3, data_gpu, labels_gpu, out_logloss, L1, L2, {fill_event3});
            logloss_event3.wait_and_throw();
            float_t logloss_down = out_logloss.to_host(this->get_queue(), {}).at(0);
            
            auto [out_derivative3, out_der_e3] = ndarray<float_t, 1>::zeros(this->get_queue(), {p_ + 1}, sycl::usm::alloc::device);
            sycl::event der_event3 = compute_logloss_with_der(this->get_queue(), params_gpu, data_gpu, labels_gpu, out_logloss, out_derivative3, L1, L2, {out_der_e3});
            der_event3.wait_and_throw();

            auto out_der_ptr3 = out_derivative3.to_host(this->get_queue(), {});

        

            REQUIRE(abs((logloss_up - logloss_down) - 2 * eps * out_derivative_host.at(i)) < 1e-5);

            for (int j = 0; j <= p_; ++j) {
                
                REQUIRE(abs((out_der_ptr2.at(j) - out_der_ptr3.at(j)) - 2 * eps * hessian_host.at(i, j)));
            }

            cur_param[i] += eps;
        }


        SUCCEED();

    }


private:

    // int64_t n_;
    // int64_t p_;


};

using logloss_types = COMBINE_TYPES((float, double));


TEMPLATE_TEST_M(logloss_test, "correlation on diagonal data", "[cor]", logloss_types) {
    this->test_gold_input();

}

} // namespace oneapi::dal::backend::primitives::test
