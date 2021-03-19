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
#include "oneapi/dal/test/engine/io.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/backend/primitives/stat/cov.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace la = te::linalg;

template <typename Float>
class cov_test : public te::float_algo_fixture<Float> {
public:
    auto allocate_arrays(std::int64_t column_count) {
        auto& q = this->get_queue();
        auto sums = ndarray<Float, 1>::empty(q, { column_count });
        auto corr = ndarray<Float, 2>::empty(q, { column_count, column_count });
        auto means = ndarray<Float, 1>::empty(q, { column_count });
        auto vars = ndarray<Float, 1>::empty(q, { column_count });
        auto tmp = ndarray<Float, 1>::empty(q, { column_count });
        return std::make_tuple(sums, corr, means, vars, tmp);
    }

    void check_correlation_for_uncorrelated_data(const ndarray<Float, 2>& corr) const {
        const auto corr_mat = la::matrix<Float>::wrap_nd(corr);
        const double eps = te::get_tolerance<Float>(1e-4, 1e-6);

        la::enumerate(corr_mat, [&](std::int64_t i, std::int64_t j, Float x) {
            if (i == j) {
                if (std::abs(x - 1.0) > eps) {
                    CAPTURE(i, j, x, eps);
                    FAIL("Unexpected diagonal element of correlation matrix");
                }
            }
            else {
                if (std::abs(x) > eps) {
                    CAPTURE(i, j, x, eps);
                    FAIL("Unexpected non-diagonal element of correlation matrix");
                }
            }
        });
    }

    void check_constant_variance(const ndarray<Float, 1>& vars,
                                 std::int64_t row_count,
                                 double expected_var) const {
        const auto vars_mat = la::matrix<Float>::wrap_nd(vars);
        const double eps = std::abs(expected_var) * te::get_tolerance_for_sum<Float>(row_count);

        la::enumerate_linear(vars_mat, [&](std::int64_t i, Float var) {
            if (std::abs(double(var) - expected_var) > eps) {
                CAPTURE(i, var, expected_var);
                FAIL("Unexpected variance");
            }
        });
    }

    void check_constant_mean(const ndarray<Float, 1>& means,
                             std::int64_t row_count,
                             double expected_mean) const {
        const auto means_mat = la::matrix<Float>::wrap_nd(means);
        const double eps = std::abs(expected_mean) * te::get_tolerance_for_sum<Float>(row_count);

        la::enumerate_linear(means_mat, [&](std::int64_t i, Float mean) {
            if (std::abs(double(mean) - expected_mean) > eps) {
                CAPTURE(i, mean, expected_mean);
                FAIL("Unexpected mean");
            }
        });
    }
};

TEMPLATE_TEST_M(cov_test, "correlation on uncorrelated data", "[cor]", float, double) {
    // DPC++ GEMM used underneath correlation is not supported on GPU
    SKIP_IF(this->get_policy().is_cpu());

    // Test takes too long time if HW emulates float64
    SKIP_IF(this->not_float64_friendly());

    const float_t diag_element = 10.5;

    // Generate dataset, where the upper square part of the matrix is diagonal
    // and the rest are zeros, for example:
    // [ x 0 0 ]
    // [ 0 x 0 ]
    // [ 0 0 x ]
    // [ 0 0 0 ]
    const auto df =
        GENERATE_DATAFRAME(te::dataframe_builder{ 1000000, 100 }.fill_diag(diag_element));

    auto [sums, corr, means, vars, tmp] = this->allocate_arrays(df.get_column_count());
    auto sums_event = sums.fill(this->get_queue(), diag_element);
    const auto data = df.get_table(this->get_policy(), this->get_homogen_table_id());

    INFO("run correlation");
    correlation(this->get_queue(), data, sums, corr, means, vars, tmp, { sums_event })
        .wait_and_throw();

    INFO("check if correlation matrix is ones")
    this->check_correlation_for_uncorrelated_data(corr);

    // The upper part of data matrix is diagonal. In diagonal matrix each column
    // contains only one non-zero element (`diag_element`), so mean and
    // variances for each feature can be computed trivially using `diag_element`
    // value.

    INFO("check if mean is expected")
    double n = df.get_row_count();
    const double expected_mean = double(diag_element) / n;
    this->check_constant_mean(means, n, expected_mean);

    INFO("check if variance is expected")
    n = df.get_row_count();
    const double d = double(diag_element) * double(diag_element);
    ONEDAL_ASSERT(n > 1);
    const double expected_var = (d - d / n) / (n - 1.0);
    this->check_constant_variance(vars, n, expected_var);
}

TEMPLATE_TEST_M(cov_test, "correlation on one-row table", "[cor]", float) {
    // DPC++ GEMM used underneath correlation is not supported on GPU
    SKIP_IF(this->get_policy().is_cpu());

    constexpr std::int64_t column_count = 3;
    const float data_ptr[column_count] = { 0.1f, 0.2f, 0.3f };
    const auto data = homogen_table::wrap(data_ptr, 1, column_count);

    auto [sums, corr, means, vars, tmp] = this->allocate_arrays(column_count);
    auto sums_event = sums.assign(this->get_queue(), data_ptr, column_count);

    INFO("run correlation");
    correlation(this->get_queue(), data, sums, corr, means, vars, tmp, { sums_event })
        .wait_and_throw();

    INFO("check if correlation matrix is ones")
    this->check_correlation_for_uncorrelated_data(corr);
}

} // namespace oneapi::dal::backend::primitives::test
