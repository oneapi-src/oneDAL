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

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/backend/primitives/stat/cov.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <typename Float>
class cov_test : public te::policy_fixture {
public:
    void check_correlation_for_uncorrelated_data(const ndarray<Float, 2>& corr) {
        const auto corr_mat =
            te::linalg::matrix<Float>::wrap(corr.get_data(),
                                            { corr.get_shape(0), corr.get_shape(1) });

        te::linalg::enumerate(corr_mat, [&](std::int64_t i, std::int64_t j, Float x) {
            if (i == j) {
                if (int(x) != 1) {
                    CAPTURE(i, j, x);
                    FAIL("Unexpected diagonal element of correlation matrix");
                }
            }
            else {
                if (int(x) != 0) {
                    CAPTURE(i, j, x);
                    FAIL("Unexpected non-diagonal element of correlation matrix");
                }
            }
        });
    }

    void check_constant_variance(const ndarray<Float, 1>& vars, double expected_var) {
        const auto vars_mat =
            te::linalg::matrix<Float>::wrap(vars.get_data(), { vars.get_count(), 1 });

        const double eps = te::get_tolerance<Float>(1e-6, 1e-12);

        te::linalg::enumerate_linear(vars_mat, [&](std::int64_t i, Float var) {
            if (std::abs(double(var) - expected_var) >= eps) {
                CAPTURE(i, var, expected_var);
                FAIL("Unexpected variance");
            }
        });
    }

    void check_constant_mean(const ndarray<Float, 1>& means, double expected_mean) {
        const auto means_mat =
            te::linalg::matrix<Float>::wrap(means.get_data(), { means.get_count(), 1 });

        const double eps = te::get_tolerance<Float>(1e-6, 1e-12);

        te::linalg::enumerate_linear(means_mat, [&](std::int64_t i, Float mean) {
            if (std::abs(double(mean) - expected_mean) >= eps) {
                CAPTURE(i, mean, expected_mean);
                FAIL("Unexpected mean");
            }
        });
    }
};

TEMPLATE_TEST_M(cov_test, "correlation on uncorrelated data", "[cor]", float, double) {
    using float_t = TestType;
    auto& queue = this->get_queue();

    const float_t diag_element = 10.5;
    const std::int64_t row_count = 5000;

    const auto df =
        GENERATE_DATAFRAME(te::dataframe_builder{ row_count, 100 }.fill_diag(diag_element));

    auto corr = ndarray<float_t, 2>::empty(queue, { 100, 100 });
    auto means = ndarray<float_t, 1>::empty(queue, { 100 });
    auto vars = ndarray<float_t, 1>::empty(queue, { 100 });
    auto tmp = ndarray<float_t, 1>::empty(queue, { 100 });

    auto [sums, sums_event] = ndarray<float_t, 1>::full(queue, { 100 }, diag_element);
    const auto data = df.get_table(this->get_policy(), te::table_id::homogen<float_t>());

    correlation(queue, data, sums, corr, means, vars, tmp, { sums_event }).wait_and_throw();

    SECTION("correlation matrix is ones") {
        this->check_correlation_for_uncorrelated_data(corr);
    }

    SECTION("mean is expected") {
        const double n = row_count;
        const double expected_mean = double(diag_element) / n;
        this->check_constant_mean(means, expected_mean);
    }

    SECTION("variance is expected") {
        const double n = row_count;
        const double d = double(diag_element) * double(diag_element);
        const double expected_var = (d - d / n) / (n - 1.0);
        this->check_constant_variance(vars, expected_var);
    }
}

} // namespace oneapi::dal::backend::primitives::test
