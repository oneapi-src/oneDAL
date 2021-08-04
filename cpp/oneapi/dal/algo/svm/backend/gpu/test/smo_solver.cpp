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

    template <typename Float>
    void print_ndview(sycl::queue& q, const pr::ndarray<Float, 1>& f, std::string str) {
        auto host_ndarr = f.to_host(q);
        const Float* f_ptr = host_ndarr.get_data();
        std::cout << "Printing: " << str << "   : ";
        for (std::int64_t i = 0; i < f.get_dimension(0); i++)
            std::cout << static_cast<float>(f_ptr[i]) << " ";
        std::cout << std::endl;
    }

    void test_smo_solver(const std::vector<Float>& x,
                         const std::vector<Float>& y,
                         const std::vector<std::uint32_t>& ws_indices,
                         const Float C,
                         const std::int64_t row_count,
                         const std::int64_t column_count) {
        auto& q = this->get_queue();

        INFO("Compute kernel function values");
        auto x_table = homogen_table::wrap(x.data(), row_count, column_count);
        const auto kernel_desc = dal::linear_kernel::descriptor{}.set_scale(1.0).set_shift(0.0);
        const auto result = dal::compute(q, kernel_desc, x_table, x_table);
        const auto kernel_values_nd =
            pr::table2ndarray_1d<Float>(q, result.get_values(), sycl::usm::alloc::device);

        INFO("Allocate ndarray");
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

        const std::int64_t max_inner_iter = 100000;
        const Float eps = 1.0e-3;
        const Float tau = 1.0e-12;

        INFO("Run solve smo");
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

        print_ndview<std::uint32_t>(q, inner_iter_count_nd, "INNER ITER COUNT: ");
        print_ndview<Float>(q, f_diff_nd, "F DIFF: ");
        print_ndview<Float>(q, alpha_nd, " ALPHA: ");
        print_ndview<Float>(q, delta_alpha_nd, " DELTA ALPHA: ");
        print_ndview<Float>(q, f_nd, " F: ");

        // INFO("Check ws_indices");
        // const auto indices_arr = ws_indices.flatten(q);

        // const auto indices_mat_host = la::matrix<std::uint32_t>::wrap(indices_arr).to_host();
        // const auto indices_arr_host = indices_mat_host.get_array();
        // for (std::int64_t i = 0; i < ws_indices.get_dimension(0); i++)
        //     REQUIRE(indices_arr_host[i] == expected_ws_indices[i]);
    }
};

using working_set_types = std::tuple<float, double>;

TEMPLATE_LIST_TEST_M(smo_solver_test,
                     "smo solver common flow",
                     "[svm][smo_solver]",
                     working_set_types) {
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
    const std::vector<float_t> y = { -1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1 };
    const std::vector<std::uint32_t> ws_indices = { 1, 2, 7, 9, 10, 11, 12, 14,
                                                    0, 3, 4, 5, 6,  8,  13, 15 };

    constexpr float_t C = 10.0;

    this->test_smo_solver(x, y, ws_indices, C, row_count, column_count);
}

// TEMPLATE_LIST_TEST_M(smo_solver_test,
//                      "not enough elements in upper set",
//                      "[svm][working_set]",
//                      working_set_types) {
//     SKIP_IF(this->get_policy().is_cpu());

//     using float_t = TestType;

//     constexpr std::int64_t row_count = 10;

//     const std::vector<float_t> f = { 10, 2, 3, 6, 9, 1, 7, 5, 4, 8 };
//     const std::vector<float_t> y = { -1, -1, -1, -1, -1, 1, 1, 1, 1, 1 };
//     const std::vector<float_t> alpha = { 0.0, 0.0, 0.0, 1.3, 0.0, 1.5, 2.0, 2.5, 2.5, 2.5 };

//     constexpr float_t C = 2.5;

//     constexpr std::int64_t expected_ws_count = 8;

//     const std::vector<std::uint32_t> expected_ws_indices = { 5, 3, 6, 8, 7, 9, 4, 0 };

//     this->test_smo_solver(f, y, alpha, C, row_count, expected_ws_count, expected_ws_indices);
// }

// TEMPLATE_LIST_TEST_M(smo_solver_test,
//                      "not enough elements in lower set",
//                      "[svm][working_set]",
//                      working_set_types) {
//     SKIP_IF(this->get_policy().is_cpu());

//     using float_t = TestType;

//     constexpr std::int64_t row_count = 10;

//     const std::vector<float_t> f = { 10, 2, 3, 6, 9, 1, 7, 5, 4, 8 };
//     const std::vector<float_t> y = { -1, -1, -1, -1, -1, 1, 1, 1, 1, 1 };
//     const std::vector<float_t> alpha = { 0.0, 2.5, 2.5, 2.5, 1.5, 0.0, 0.0, 0.0, 2.5, 0.0 };

//     constexpr float_t C = 2.5;

//     constexpr std::int64_t expected_ws_count = 8;

//     const std::vector<std::uint32_t> expected_ws_indices = { 5, 1, 2, 7, 8, 4, 0, 3 };

//     this->test_smo_solver(f, y, alpha, C, row_count, expected_ws_count, expected_ws_indices);
// }

} // namespace oneapi::dal::svm::backend::test