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
class smo_solver_test : public te::float_algo_fixture<TestType> {
public:
    using float_t = TestType;

    void test_smo_solver(const std::vector<float_t>& x,
                         const std::vector<float_t>& y,
                         const float_t C,
                         const float_t eps,
                         const std::int64_t max_inner_iter,
                         const std::int64_t row_count,
                         const std::int64_t column_count,
                         const std::int64_t expected_inner_iter_count,
                         const float_t expected_f_diff,
                         const std::vector<float_t>& expected_alpha,
                         const std::vector<float_t>& expected_delta_alpha) {
        auto& q = this->get_queue();

        INFO("compute kernel function values");
        auto x_table = homogen_table::wrap(x.data(), row_count, column_count);
        const auto kernel_desc = dal::linear_kernel::descriptor{}.set_scale(1.0).set_shift(0.0);
        const auto result = dal::compute(q, kernel_desc, x_table, x_table);
        const auto kernel_values_nd =
            pr::table2ndarray<float_t>(q, result.get_values(), sycl::usm::alloc::device);

        INFO("allocate ndarray");
        auto y_host_nd = pr::ndarray<float_t, 1>::wrap(y.data(), row_count);
        auto y_nd = y_host_nd.to_device(q);

        auto ws_indices_nd =
            pr::ndarray<std::int32_t, 1>::empty(q, { row_count }, sycl::usm::alloc::device);
        auto arange_event = ws_indices_nd.arange(q);

        auto f_nd = pr::ndarray<float_t, 1>::empty(q, { row_count }, sycl::usm::alloc::device);
        auto invert_y_event = invert_values(q, y_nd, f_nd);

        auto [alpha_nd, alpha_event] =
            pr::ndarray<float_t, 1>::zeros(q, { row_count }, sycl::usm::alloc::device);
        auto delta_alpha_nd =
            pr::ndarray<float_t, 1>::empty(q, { row_count }, sycl::usm::alloc::device);
        auto f_diff_nd = pr::ndarray<float_t, 1>::empty(q, { 1 }, sycl::usm::alloc::device);
        auto inner_iter_count_nd =
            pr::ndarray<std::int32_t, 1>::empty(q, { 1 }, sycl::usm::alloc::device);

        const float_t tau = 1.0e-12;

        INFO("run solve smo");
        solve_smo<float_t>(q,
                           kernel_values_nd,
                           ws_indices_nd,
                           y_nd,
                           f_nd,
                           row_count,
                           row_count,
                           max_inner_iter,
                           C,
                           eps,
                           tau,
                           alpha_nd,
                           delta_alpha_nd,
                           f_diff_nd,
                           inner_iter_count_nd,
                           { invert_y_event, alpha_event, arange_event })
            .wait_and_throw();

        INFO("check if alpha is expected");
        check_ndarray(alpha_nd, expected_alpha);

        INFO("check if delta alpha is expected");
        check_ndarray(delta_alpha_nd, expected_delta_alpha);

        INFO("check if f diff is expected");
        check_f_diff(f_diff_nd, expected_f_diff);

        INFO("check if inner iter count is expected");
        check_inner_iter_count(inner_iter_count_nd, expected_inner_iter_count);
    }

    void check_ndarray(const pr::ndarray<float_t, 1>& res,
                       const std::vector<float_t>& expected_res) {
        auto row_count = res.get_count();

        const auto res_host = res.to_host(this->get_queue());
        auto expected_res_host = pr::ndarray<float_t, 1>::wrap(expected_res.data(), row_count);
        const float_t* res_ptr = res_host.get_data();
        const float_t* expected_res_ptr = expected_res_host.get_data();

        for (std::int64_t el = 0; el < row_count; el++) {
            REQUIRE(fabs(res_ptr[el] - expected_res_ptr[el]) < 1.0e-3);
        }
    }

    void check_f_diff(const pr::ndarray<float_t, 1>& f_diff_nd, const float_t expected_f_diff) {
        auto& q = this->get_queue();

        const auto f_diff_arr = f_diff_nd.flatten(q);
        const auto f_diff_mat_host = la::matrix<float_t>::wrap(f_diff_arr).to_host();
        const auto f_diff_arr_host = f_diff_mat_host.get_array();

        REQUIRE(fabs(expected_f_diff - f_diff_arr_host[0]) < 1.0e-3);
    }

    void check_inner_iter_count(const pr::ndarray<std::int32_t, 1>& inner_iter_count_nd,
                                const std::int64_t expected_inner_iter_count) {
        auto& q = this->get_queue();

        const auto inner_iter_count_arr = inner_iter_count_nd.flatten(q);
        const auto inner_iter_count_mat_host =
            la::matrix<std::int32_t>::wrap(inner_iter_count_arr).to_host();
        const auto inner_iter_count_arr_host = inner_iter_count_mat_host.get_array();

        REQUIRE(expected_inner_iter_count >= inner_iter_count_arr_host[0]);
    }
};

using smo_solver_types = std::tuple<float, double>;

TEMPLATE_LIST_TEST_M(smo_solver_test,
                     "smo solver common flow",
                     "[svm][smo_solver]",
                     smo_solver_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

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

    constexpr float_t C = 10.0;
    constexpr float_t eps = 1.0e-3;
    constexpr std::int64_t max_inner_iter = 100;
    constexpr std::int64_t expected_inner_iter_count = 5;
    constexpr float_t expected_f_diff = 2.0;

    const std::vector<float_t> expected_alpha = { 10, 0,  0,  10, 10, 0, 10, 0,
                                                  10, 10, 10, 10, 10, 0, 0,  10 };

    const std::vector<float_t> expected_delta_alpha = { 10,  0,  0,  -10, -10, -0, -10, 0,
                                                        -10, 10, 10, 10,  10,  -0, 0,   -10 };

    this->test_smo_solver(x,
                          y,
                          C,
                          eps,
                          max_inner_iter,
                          row_count,
                          column_count,
                          expected_inner_iter_count,
                          expected_f_diff,
                          expected_alpha,
                          expected_delta_alpha);
}

TEMPLATE_LIST_TEST_M(smo_solver_test,
                     "smo solver with bigger number of interations",
                     "[svm][smo_solver]",
                     smo_solver_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

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

    constexpr float_t C = 3.0;
    constexpr float_t eps = 1.0e-3;
    constexpr std::int64_t max_inner_iter = 1000;
    constexpr std::int64_t expected_inner_iter_count = 15;
    constexpr float_t expected_f_diff = 2.0;

    const std::vector<float_t> expected_alpha = { 0, 0, 0, 0, 0.931198, 0.584559, 0, 3,
                                                  3, 0, 0, 3, 0.14977,  0,        0, 2.50359 };

    const std::vector<float_t> expected_delta_alpha = { 0,  0,       0, -0, -0.931198, 0.584559, 0,
                                                        3,  3,       0, 0,  -3,        -0.14977, -0,
                                                        -0, -2.50359 };

    this->test_smo_solver(x,
                          y,
                          C,
                          eps,
                          max_inner_iter,
                          row_count,
                          column_count,
                          expected_inner_iter_count,
                          expected_f_diff,
                          expected_alpha,
                          expected_delta_alpha);
}

TEMPLATE_LIST_TEST_M(smo_solver_test,
                     "smo solver with big C",
                     "[svm][smo_solver]",
                     smo_solver_types) {
    SKIP_IF(this->get_policy().is_cpu());
    SKIP_IF(this->not_float64_friendly());

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

    constexpr float_t C = 100.0;
    constexpr float_t eps = 1.0e-3;
    constexpr std::int64_t max_inner_iter = 1000;
    constexpr std::int64_t expected_inner_iter_count = 117;
    constexpr float_t expected_f_diff = 2.0;

    const std::vector<float_t> expected_alpha = { 0,   0, 76.8432, 23.1569, 0, 0, 0, 0,
                                                  100, 0, 0,       0,       0, 0, 0, 0 };

    const std::vector<float_t> expected_delta_alpha = { 0,   0, -76.8432, -23.1569, -0, -0, 0,  -0,
                                                        100, 0, 0,        0,        -0, -0, -0, 0 };

    this->test_smo_solver(x,
                          y,
                          C,
                          eps,
                          max_inner_iter,
                          row_count,
                          column_count,
                          expected_inner_iter_count,
                          expected_f_diff,
                          expected_alpha,
                          expected_delta_alpha);
}

} // namespace oneapi::dal::svm::backend::test
