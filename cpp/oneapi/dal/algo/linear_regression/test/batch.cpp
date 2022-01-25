/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#include <vector>

#include "oneapi/dal/algo/linear_regression/common.hpp"
#include "oneapi/dal/algo/linear_regression/train.hpp"
#include "oneapi/dal/algo/linear_regression/infer.hpp"

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/detail/table_builder.hpp"

#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

#include "oneapi/dal/test/engine/metrics/regression.hpp"

namespace oneapi::dal::linear_regression::test {

namespace te = dal::test::engine;
namespace de = dal::detail;
namespace la = te::linalg;

template <typename TestType>
class lr_batch_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using float_t = std::tuple_element_t<0, TestType>;
    using method_t = std::tuple_element_t<1, TestType>;
    using task_t = std::tuple_element_t<2, TestType>;

    void generate_dimensions() {
        s_count_ = GENERATE(111, 113);
        f_count_ = GENERATE(2, 3, 5);
        r_count_ = GENERATE(2, 7, 9);
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<float_t>();
    }

    table compute_responses(const table& beta, const table& bias, const table& data) const {
        auto res_arr = array<float_t>::zeros(this->s_count_ * this->r_count_);
        const auto beta_arr = row_accessor<const float_t>(beta).pull({ 0, -1 });
        const auto bias_arr = row_accessor<const float_t>(bias).pull({ 0, -1 });
        const auto data_arr = row_accessor<const float_t>(data).pull({ 0, -1 });

        for (std::int64_t s = 0; s < this->s_count_; ++s) {
            for (std::int64_t r = 0; r < this->r_count_; ++r) {
                for (std::int64_t f = 0; f < this->f_count_; ++f) {
                    const auto& v = data_arr[s * this->f_count_ + f];
                    const auto& b = beta_arr[f * this->r_count_ + r];
                    *(res_arr.get_mutable_data() + s * this->r_count_ + r) += v * b;
                }
            }
        }

        for (std::int64_t s = 0; s < this->s_count_; ++s) {
            for (std::int64_t r = 0; r < this->r_count_; ++r) {
                *(res_arr.get_mutable_data() + s * this->r_count_ + r) += bias_arr[r];
            }
        }

        return homogen_table::wrap(res_arr, this->s_count_, this->r_count_);
    }

    std::tuple<table, table> generate_betas() const {
        std::tuple<table, table> result{ {}, {} };
        const auto betas_dataframe = GENERATE_DATAFRAME(
            te::dataframe_builder{ this->f_count_, this->r_count_ }.fill_uniform(-10.1, 10.1));
        std::get<0>(result) = betas_dataframe.get_table(this->get_homogen_table_id());
        if (this->intercept_) {
            const auto bias_dataframe = GENERATE_DATAFRAME(
                te::dataframe_builder{ std::int64_t(1), this->r_count_ }.fill_uniform(-15.5, 15.5));
            std::get<1>(result) = betas_dataframe.get_table(this->get_homogen_table_id());
        }
        else {
            auto bias_arr = array<float_t>::zeros(this->r_count_);
            std::get<1>(result) = homogen_table::wrap(bias_arr, std::int64_t(1), this->r_count_);
        }
        return result;
    }

    void check_table_dimensions() {
        REQUIRE(this->x_train_.get_column_count() == this->f_count_);
        REQUIRE(this->x_train_.get_row_count() == this->s_count_);
        REQUIRE(this->x_test_.get_column_count() == this->f_count_);
        REQUIRE(this->x_test_.get_row_count() == this->s_count_);
        REQUIRE(this->y_train_.get_column_count() == this->r_count_);
        REQUIRE(this->y_train_.get_row_count() == this->s_count_);
        REQUIRE(this->y_test_.get_column_count() == this->r_count_);
        REQUIRE(this->y_test_.get_row_count() == this->s_count_);
    }

    void generate() {
        this->generate_dimensions();
        auto [beta, bias] = generate_betas();

        const auto train_dataframe = GENERATE_DATAFRAME(
            te::dataframe_builder{ this->s_count_, this->f_count_ }.fill_uniform(-5.5, 3.5));
        this->x_train_ = train_dataframe.get_table(this->get_homogen_table_id());

        const auto test_dataframe = GENERATE_DATAFRAME(
            te::dataframe_builder{ this->s_count_, this->f_count_ }.fill_uniform(-7.5, 5.5));
        this->x_test_ = test_dataframe.get_table(this->get_homogen_table_id());

        this->y_train_ = compute_responses(beta, bias, this->x_train_);
        this->y_test_ = compute_responses(beta, bias, this->x_test_);

        this->check_table_dimensions();
    }

    auto get_descriptor() const {
        return linear_regression::descriptor<float_t, method_t, task_t>();
    }

    void check_results(const infer_result<>& res, double tol = 1e-5) {
        const table& gtr_table = this->y_test_;
        const table& res_table = res.get_responses();

        const table scr_table = te::mse_score<float_t>(res_table, gtr_table);

        const auto score = row_accessor<const float_t>(scr_table).pull({ 0, -1 });

        for (std::int64_t r = 0; r < this->r_count_; ++r) {
            REQUIRE(score[r] < tol);
        }
    }

    void run_and_check() {
        const auto desc = this->get_descriptor();
        const auto train_res = this->train(desc, this->x_train_, this->y_train_);
        const auto infer_res = this->infer(desc, this->x_test_, train_res.get_model());

        check_results(infer_res);
    }

private:
    bool intercept_ = true;
    std::int64_t s_count_;
    std::int64_t f_count_;
    std::int64_t r_count_;

    table x_test_;
    table y_test_;
    table x_train_;
    table y_train_;
};

using lr_types = COMBINE_TYPES((float, double),
                               (linear_regression::method::norm_eq),
                               (linear_regression::task::regression));

TEMPLATE_LIST_TEST_M(lr_batch_test, "LR common flow", "[lr][batch]", lr_types) {
    //SKIP_IF(this->get_policy().is_gpu());
    SKIP_IF(this->not_float64_friendly());

    this->generate();

    this->run_and_check();
}

} // namespace oneapi::dal::linear_regression::test
