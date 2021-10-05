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
#include "oneapi/dal/test/engine/io.hpp"
#include "oneapi/dal/test/engine/math.hpp"

#include "oneapi/dal/backend/primitives/utils.hpp"

#include "oneapi/dal/algo/svm/backend/gpu/smo_solver.hpp"
#include "oneapi/dal/algo/linear_kernel/compute.hpp"

namespace oneapi::dal::svm::backend::test {

namespace pr = dal::backend::primitives;
namespace te = dal::test::engine;
namespace la = te::linalg;

template <typename TestType>
class smo_solver_test : public te::policy_fixture {
public:
    using Float = TestType;

    void test_smo_solver(const std::vector<Float>& x,
                         const std::vector<Float>& y,
                         const std::vector<std::uint32_t>& ws_indices,
                         const Float C,
                         const Float eps,
                         const std::int64_t max_inner_iter,
                         const std::int64_t row_count,
                         const std::int64_t column_count,
                         const std::int64_t expected_inner_iter_count,
                         const Float expected_f_diff,
                         const Float expected_objective_func) {
        auto& q = this->get_queue();

        INFO("compute kernel function values");
        auto x_table = homogen_table::wrap(x.data(), row_count, column_count);
        const auto kernel_desc = dal::linear_kernel::descriptor{}.set_scale(1.0).set_shift(0.0);
        const auto result = dal::compute(q, kernel_desc, x_table, x_table);
        const auto kernel_values_nd =
            pr::table2ndarray<Float>(q, result.get_values(), sycl::usm::alloc::device);

        INFO("allocate ndarray");
        auto y_host_nd = pr::ndarray<Float, 1>::wrap(y.data(), row_count);
        auto y_nd = y_host_nd.to_device(q);

        auto ws_indices_host_nd = pr::ndarray<std::uint32_t, 1>::wrap(ws_indices.data(), row_count);
        auto ws_indices_nd = ws_indices_host_nd.to_device(q);

        auto f_nd = pr::ndarray<Float, 1>::empty(q, { row_count }, sycl::usm::alloc::device);
        auto invert_y_event = invert_values(q, y_nd, f_nd);

        auto alpha_nd = pr::ndarray<Float, 1>::empty(q, { row_count }, sycl::usm::alloc::device);
        auto delta_alpha_nd =
            pr::ndarray<Float, 1>::empty(q, { row_count }, sycl::usm::alloc::device);
        auto f_diff_nd = pr::ndarray<Float, 1>::empty(q, { 1 }, sycl::usm::alloc::device);
        auto inner_iter_count_nd =
            pr::ndarray<std::uint32_t, 1>::empty(q, { 1 }, sycl::usm::alloc::device);

        const Float tau = 1.0e-12;

        INFO("run solve smo");
        solve_smo<Float>(q,
                         kernel_values_nd,
                         ws_indices_nd,
                         y_nd,
                         row_count,
                         row_count,
                         max_inner_iter,
                         C,
                         eps,
                         tau,
                         alpha_nd,
                         delta_alpha_nd,
                         f_nd,
                         f_diff_nd,
                         inner_iter_count_nd,
                         { invert_y_event })
            .wait_and_throw();

        INFO("check if objective function is expected");
        check_objective_function(y, alpha_nd, f_nd, expected_objective_func);

        INFO("check if f diff is expected");
        check_f_diff(f_diff_nd, expected_f_diff);

        INFO("check if inner iter count is expected");
        check_inner_iter_count(inner_iter_count_nd, expected_inner_iter_count);
    }

    void check_objective_function(const std::vector<Float>& y,
                                  const pr::ndarray<Float, 1>& alpha_nd,
                                  const pr::ndarray<Float, 1>& f_nd,
                                  const Float expected_objective_func) {
        auto& q = this->get_queue();

        const auto alpha_arr = alpha_nd.flatten(q);
        const auto alpha_mat_host = la::matrix<Float>::wrap(alpha_arr).to_host();
        const auto alpha_arr_host = alpha_mat_host.get_array();

        const auto f_arr = f_nd.flatten(q);
        const auto f_mat_host = la::matrix<Float>::wrap(f_arr).to_host();
        const auto f_arr_host = f_mat_host.get_array();

        Float objective = 0;

        for (std::int64_t i = 0; i < alpha_nd.get_count(); i++) {
            objective += alpha_arr_host[i] - (f_arr_host[i] + y[i]) * alpha_arr_host[i] * y[i] / 2;
        }

        REQUIRE(fabs(expected_objective_func + objective) < 1e-3);
    }

    void check_f_diff(const pr::ndarray<Float, 1>& f_diff_nd, const Float expected_f_diff) {
        auto& q = this->get_queue();

        const auto f_diff_arr = f_diff_nd.flatten(q);
        const auto f_diff_mat_host = la::matrix<Float>::wrap(f_diff_arr).to_host();
        const auto f_diff_arr_host = f_diff_mat_host.get_array();

        REQUIRE(fabs(expected_f_diff - f_diff_arr_host[0]) < 1e-3);
    }

    void check_inner_iter_count(const pr::ndarray<std::uint32_t, 1>& inner_iter_count_nd,
                                const std::int64_t expected_inner_iter_count) {
        auto& q = this->get_queue();

        const auto inner_iter_count_arr = inner_iter_count_nd.flatten(q);
        const auto inner_iter_count_mat_host =
            la::matrix<std::uint32_t>::wrap(inner_iter_count_arr).to_host();
        const auto inner_iter_count_arr_host = inner_iter_count_mat_host.get_array();

        REQUIRE(expected_inner_iter_count == inner_iter_count_arr_host[0]);
    }
};

using smo_solver_types = std::tuple<float, double>;

TEMPLATE_LIST_TEST_M(smo_solver_test,
                     "smo solver common flow",
                     "[svm][smo_solver]",
                     smo_solver_types) {
    SKIP_IF(this->get_policy().is_cpu());

    using float_t = TestType;

    constexpr std::int64_t row_count = 16;
    constexpr std::int64_t column_count = 2;

    const std::vector<float_t> x = {
        -0.08311833, -0.09168183, 0.13301689,  0.02445668,  0.19215095,  -0.21031374, -0.15119842,
        -0.08235691, -0.07523747, 0.10911487,  -0.19196572, -0.00136943, -0.07403088, 0.08018183,
        0.17566872,  -0.10336669, -0.12230175, 0.10784139,  0.15150517,  0.13519147,  0.05096889,
        -0.12530502, 0.08796324,  0.1235733,   -0.02085915, 0.02664193,  -0.13818114, 0.08816879,
        0.13908519,  -0.15868474, -0.11105107, -0.07440591
    };
    const std::vector<float_t> y = { 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1 };
    const std::vector<std::uint32_t> ws_indices = { 0, 1, 2,  3,  4,  5,  6,  7,
                                                    8, 9, 10, 11, 12, 13, 14, 15 };

    constexpr float_t C = 10.0;
    constexpr float_t eps = 1.0e-3;
    constexpr std::int64_t max_inner_iter = 100;
    constexpr std::int64_t expected_inner_iter_count = 5;
    constexpr float_t expected_f_diff = 2.0;
    constexpr float_t expected_objective_func = -73.801;

    this->test_smo_solver(x,
                          y,
                          ws_indices,
                          C,
                          eps,
                          max_inner_iter,
                          row_count,
                          column_count,
                          expected_inner_iter_count,
                          expected_f_diff,
                          expected_objective_func);
}

TEMPLATE_LIST_TEST_M(smo_solver_test,
                     "smo solver with bigger number of interations",
                     "[svm][smo_solver]",
                     smo_solver_types) {
    SKIP_IF(this->get_policy().is_cpu());

    using float_t = TestType;

    constexpr std::int64_t row_count = 16;
    constexpr std::int64_t column_count = 2;

    const std::vector<float_t> x = {
        0.65087762,  1.06568966,  -1.1987315,  0.94108574,  1.19831325,  0.98786829,  0.71698069,
        -1.78011614, -1.22913346, -0.78381204, -0.74056676, 0.50802431,  -1.05146513, 1.01241273,
        0.61613063,  -1.27132561, 0.69326232,  0.04276818,  0.15616993,  1.16919702,  0.72816868,
        1.32267897,  0.96634166,  -0.70945501, -1.52944351, -0.83255698, -0.81554704, -0.88040619,
        -1.67517968, -1.82890019, 0.79891573,  -0.7414803,
    };
    const std::vector<float_t> y = {
        1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1,
    };
    const std::vector<std::uint32_t> ws_indices = { 0, 1, 2,  3,  4,  5,  6,  7,
                                                    8, 9, 10, 11, 12, 13, 14, 15 };

    constexpr float_t C = 3.0;
    constexpr float_t eps = 1.0e-3;
    constexpr std::int64_t max_inner_iter = 1000;
    constexpr std::int64_t expected_inner_iter_count = 17;
    constexpr float_t expected_f_diff = 2.0;
    constexpr float_t expected_objective_func = -12.1322;

    this->test_smo_solver(x,
                          y,
                          ws_indices,
                          C,
                          eps,
                          max_inner_iter,
                          row_count,
                          column_count,
                          expected_inner_iter_count,
                          expected_f_diff,
                          expected_objective_func);
}

TEMPLATE_LIST_TEST_M(smo_solver_test,
                     "smo solver with big C",
                     "[svm][smo_solver]",
                     smo_solver_types) {
    SKIP_IF(this->get_policy().is_cpu());

    using float_t = TestType;

    constexpr std::int64_t row_count = 16;
    constexpr std::int64_t column_count = 2;

    const std::vector<float_t> x = {
        -0.0826849,  2.16909461,  1.40344666,  0.99366156,  1.53758353,  -0.94500719, -1.71795443,
        -0.3185163,  0.89011269,  -1.03534028, -1.31709552, -1.26831733, -1.09203724, 1.45971573,
        1.09318071,  -0.90639048, 0.80109734,  -0.709544,   -0.84044139, 0.65304747,  0.77449962,
        1.37499061,  -0.01058333, 2.55043074,  -1.27203048, -0.65009482, 1.12823908,  -0.90640686,
        -0.82022575, -1.57899185, 1.39052325,  -0.28958642,
    };
    const std::vector<float_t> y = { 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1 };
    const std::vector<std::uint32_t> ws_indices = { 0, 1, 2,  3,  4,  5,  6,  7,
                                                    8, 9, 10, 11, 12, 13, 14, 15 };

    constexpr float_t C = 100.0;
    constexpr float_t eps = 1.0e-3;
    constexpr std::int64_t max_inner_iter = 117;
    constexpr std::int64_t expected_inner_iter_count = 117;
    constexpr float_t expected_f_diff = 2.0;
    constexpr float_t expected_objective_func = -157.638;

    this->test_smo_solver(x,
                          y,
                          ws_indices,
                          C,
                          eps,
                          max_inner_iter,
                          row_count,
                          column_count,
                          expected_inner_iter_count,
                          expected_f_diff,
                          expected_objective_func);
}

} // namespace oneapi::dal::svm::backend::test
