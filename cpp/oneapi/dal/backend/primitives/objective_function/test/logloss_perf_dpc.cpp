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

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <ndorder order>
struct order_tag {
    static constexpr ndorder value = order;
};

using c_order = order_tag<ndorder::c>;
using f_order = order_tag<ndorder::f>;

template <typename Param>
class logloss_perf_test : public te::float_algo_fixture<Param> {
public:
    using float_t = Param;

    void generate_input(std::int64_t n, std::int64_t p) {
        this->n_ = n;
        this->p_ = p;

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

    void measure_time() {
        constexpr float_t L1 = 1.2;
        constexpr float_t L2 = 0.7;

        auto data_array = row_accessor<const float_t>{ this->data_ }.pull(this->get_queue());
        auto data_host = ndarray<float_t, 2>::wrap(data_array.get_data(), { n_, p_ });

        auto param_array = row_accessor<const float_t>{ this->params_ }.pull(this->get_queue());
        auto params_host = ndarray<float_t, 1>::wrap(param_array.get_data(), { p_ + 1 });

        auto data_gpu = data_host.to_device(this->get_queue());
        auto labels_gpu = this->labels_.to_device(this->get_queue());
        auto params_gpu = params_host.to_device(this->get_queue());

        auto out_predictions =
            ndarray<float_t, 1>::empty(this->get_queue(), { n_ }, sycl::usm::alloc::device);

        auto p_event =
            compute_probabilities(this->get_queue(), params_gpu, data_gpu, out_predictions, {});
        p_event.wait_and_throw();

        auto out_logloss =
            ndarray<float_t, 1>::empty(this->get_queue(), { 1 }, sycl::usm::alloc::device);

        auto out_derivative =
            ndarray<float_t, 1>::empty(this->get_queue(), { p_ + 1 }, sycl::usm::alloc::device);

        BENCHMARK("Derivative computation") {
            auto fill_event1 = fill<float_t>(this->get_queue(), out_logloss, float_t(0), {});
            auto fill_event2 = fill<float_t>(this->get_queue(), out_derivative, float_t(0), {});

            auto logloss_event_der = compute_logloss_with_der(this->get_queue(),
                                                              params_gpu,
                                                              data_gpu,
                                                              labels_gpu,
                                                              out_predictions,
                                                              out_logloss,
                                                              out_derivative,
                                                              L1,
                                                              L2,
                                                              { fill_event1, fill_event2 });
            logloss_event_der.wait_and_throw();
        };

        auto out_hessian = ndarray<float_t, 2>::empty(this->get_queue(),
                                                      { p_ + 1, p_ + 1 },
                                                      sycl::usm::alloc::device);

        BENCHMARK("Hessian computation") {
            auto fill_event = fill<float_t>(this->get_queue(), out_hessian, float_t(0), {});
            auto hess_event = compute_hessian(this->get_queue(),
                                              params_gpu,
                                              data_gpu,
                                              labels_gpu,
                                              out_predictions,
                                              out_hessian,
                                              L1,
                                              L2,
                                              { fill_event });
            hess_event.wait_and_throw();
        };
    }

private:
    std::int64_t n_;
    std::int64_t p_;
    table data_;
    table params_;
    ndarray<std::int32_t, 1> labels_;
};

TEMPLATE_TEST_M(logloss_perf_test, "perfomance test square", "[logloss][5000*5000]", double) {
    SKIP_IF(this->not_float64_friendly());
    this->generate_input(5000, 5000);
    this->measure_time();
}

TEMPLATE_TEST_M(logloss_perf_test, "perfomance test small p", "[logloss][10000*100]", double) {
    SKIP_IF(this->not_float64_friendly());
    this->generate_input(100000, 100);
    this->measure_time();
}

TEMPLATE_TEST_M(logloss_perf_test, "perfomance test small n", "[logloss][100 * 1000]", double) {
    SKIP_IF(this->not_float64_friendly());
    this->generate_input(100, 7000);
    this->measure_time();
}

} // namespace oneapi::dal::backend::primitives::test
