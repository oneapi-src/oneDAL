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

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/regression.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"

#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace de = dal::detail;
namespace la = te::linalg;

class uniform_test : public te::policy_fixture {
public:
    using res_t = float;

    auto compute_results(const ndview<res_t, 2>& responses) {
        const auto r = responses.get_dimension(0);
        const auto k = responses.get_dimension(1);
        auto res = ndarray<res_t, 1>::empty(this->get_queue(), { r });
        for (std::int32_t i = 0; i < r; ++i) {
            res_t acc = 0;
            for (std::int32_t j = 0; j < k; ++j) {
                acc += responses.at(i, j);
            }
            res.at(i) = acc / double(k);
        }
        return res;
    }

    void test_correctness(const ndview<res_t, 2>& responses, const ndview<res_t, 1>& results) {
        const auto gtr_res = compute_results(responses);
        const auto r = results.get_dimension(0);
        for (std::int32_t i = 0; i < r; ++i) {
            const auto val = results.at(i);
            const auto gtr = gtr_res.at(i);
            const auto dif = std::abs(val - gtr);
            REQUIRE(dif < 1.e-6);
        }
    }

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<res_t>();
    }

    auto generate_input(std::int32_t m, std::int32_t n) {
        const auto train_dataframe =
            GENERATE_DATAFRAME(te::dataframe_builder{ m, n }.fill_uniform(-0.2, 0.5));
        this->input_table_ = train_dataframe.get_table(this->get_homogen_table_id());

        return table2ndarray<res_t>(this->get_queue(), this->input_table_);
    }

    auto generate_input() {
        return generate_input(m_, n_);
    }

    void generate() {
        this->m_ = GENERATE(1, 16, 128, 1024);
        this->n_ = GENERATE(1, 16, 128, 1024);
    }

    void test_pipeline() {
        auto x = this->generate_input();
        auto y = ndarray<res_t, 1>::empty(this->get_queue(), { m_ });

        auto regression = make_uniform_regression<res_t>(this->get_queue(), m_, n_);

        regression->operator()(x, y).wait_and_throw();

        test_correctness(x, y);
    }

private:
    std::int64_t m_, n_;
    table input_table_;
};

TEST_M(uniform_test, "uniform regression - naive", "[ndarray]") {
    this->generate();
    this->test_pipeline();
}

} // namespace oneapi::dal::backend::primitives::test
