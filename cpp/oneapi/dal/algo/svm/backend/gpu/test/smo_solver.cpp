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

        // const std::vector<Float> k_ws  = {-0.0132984, 0.0182916, 0.0204157, -0.0221261, -0.00733927, -0.0255682, -0.00788638, 0.0208389, -0.0136308, 0.0234591, 0.00371518, 0.0147228, -0.00212305, -0.0162241, 0.0146198, -0.0165914, 0.00331068, 0.0204157, 0.0811539, -0.0117321, -0.0374053, -0.0365984, -0.0310884, 0.0554943, -0.0461809, 0.000679239, 0.0361471, -0.00908694, -0.00961127, -0.0450947, 0.0600989, -0.00568998, -0.00512444, 0.0208389, 0.0554943, -0.0180479, -0.0244957, -0.0335808, -0.021293, 0.0415442, -0.0326318, 0.0126404, 0.021906, 0.00267903, -0.00641819, -0.0333878, 0.0408356, -0.0118171, -0.0249875, 0.0234591, 0.000679239, -0.0340413, 0.00335253, -0.0292689, -0.000376163, 0.0126404, -0.00395011, 0.0412306, -0.00921812, 0.0300329, 0.000441492, -0.00901549, -0.000380699, -0.0268839, 0.00725175, 0.00371518, 0.0361471, 0.00261332, -0.0175074, -0.00961268, -0.0138205, 0.021906, -0.0197467, -0.00921812, 0.0182992, -0.011001, -0.00440154, -0.0180909, 0.026973, 0.00366328, -0.0186408, 0.0147228, -0.00908694, -0.023477, 0.00686555, -0.0170552, 0.00339634, 0.00267903, 0.00256826, 0.0300329, -0.011001, 0.0230079, 0.00145739, -0.00125955, -0.00737481, -0.018963, -0.000708803, -0.00212305, -0.00961127, 0.000959724, 0.00447642, 0.00396776, 0.00368042, -0.00641819, 0.00542421, 0.000441492, -0.00440154, 0.00145739, 0.0011449, 0.00523133, -0.00712887, 0.000334114, 0.00298798, 0.0146198, 0.0600989, -0.00796068, -0.0277793, -0.0264823, -0.0230202, 0.0408356, -0.0341231, -0.000380699, 0.026973, -0.00737481, -0.00712887, -0.03321, 0.0445255, -0.00363848, 0.0153142, -0.0132984, 0.00331068, 0.020118, -0.00375024, 0.0160814, -0.00119789, -0.00512444, 0.000278421, -0.0249875, 0.00725175, -0.0186408, -0.000708803, 0.00340191, 0.00298798, 0.016052, 0.020118, -0.0221261, -0.0117321, 0.0296436, 0.00238942, 0.0291377, 0.00458983, -0.0180479, 0.00961035, -0.0340413, 0.00261332, -0.023477, 0.000959724, 0.0136315, -0.00796068, 0.0229186, -0.00375024, -0.00733927, -0.0374053, 0.00238942, 0.0175667, 0.0142936, 0.0143189, -0.0244957, 0.0209688, 0.00335253, -0.0175074, 0.00686555, 0.00447642, 0.0200169, -0.0277793, 0.00023641, 0.0160814, -0.0255682, -0.0365984, 0.0291377, 0.0142936, 0.0368527, 0.0141016, -0.0335808, 0.0233301, -0.0292689, -0.00961268, -0.0170552, 0.00396776, 0.0264053, -0.0264823, 0.0214199, -0.00119789, -0.00788638, -0.0310884, 0.00458983, 0.0143189, 0.0141016, 0.0119097, -0.021293, 0.017701, -0.000376163, -0.0138205, 0.00339634, 0.00368042, 0.0172992, -0.0230202, 0.00225521, 0.000278421, -0.0136308, -0.0461809, 0.00961035, 0.0209688, 0.0233301, 0.017701, -0.0326318, 0.0265875, -0.00395011, -0.0197467, 0.00256826, 0.00542421, 0.026408, -0.0341231, 0.0055577, 0.00340191, -0.0162241, -0.0450947, 0.0136315, 0.0200169, 0.0264053, 0.0172992, -0.0333878, 0.026408, -0.00901549, -0.0180909, -0.00125955, 0.00523133, 0.0268678, -0.03321, 0.00878488, 0.016052, -0.0165914, -0.00568998, 0.0229186, 0.00023641, 0.0214199, 0.00225521, -0.0118171, 0.0055577, -0.0268839, 0.00366328, -0.018963, 0.000334114, 0.00878488, -0.00363848, 0.0178686 };
        // auto k_ws_host =  pr::ndarray<Float, 1>::wrap(k_ws.data(), row_count * row_count); 
        // auto kernel_values_nd = k_ws_host.to_device(q);

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

        const std::int64_t max_inner_iter = 1;
        const Float eps = 1.0e-3;
        const Float tau = 1.0e-12;

        print_ndview<Float>(q, f_nd, "F: ");

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

        // print_ndview<Float>(q, kernel_values_nd, "KERNEL VAL: ");

        print_ndview<std::uint32_t>(q, inner_iter_count_nd, "INNER ITER COUNT: ");
        print_ndview<Float>(q, f_diff_nd, "F DIFF: ");
        print_ndview<Float>(q, alpha_nd, " ALPHA: ");
        print_ndview<Float>(q, delta_alpha_nd, " DELTA ALPHA: ");
        print_ndview<Float>(q, f_nd, " F: ");

        INFO("Compute obj function");
        const auto alpha_arr = alpha_nd.flatten(q);
        const auto alpha_mat_host = la::matrix<Float>::wrap(alpha_arr).to_host();
        const auto alpha_arr_host = alpha_mat_host.get_array();

        const auto f_arr = f_nd.flatten(q);
        const auto f_mat_host = la::matrix<Float>::wrap(f_arr).to_host();
        const auto f_arr_host = f_mat_host.get_array();

        Float objective = 0;

        for (std::int64_t i = 0; i < row_count; i++) {
            objective += alpha_arr_host[i] - (f_arr_host[i] + y[i]) * alpha_arr_host[i] * y[i] / 2;
        }
        std::cout << "OBJECTIVE RES: " << -objective << std::endl;
        // for (std::int64_t i = 0; i < ws_indices.g et_dimension(0); i++)
            // REQUIRE(indices_arr_host[i] == expected_ws_indices[i]);
    }
};

// using working_set_types = std::tuple<float, double>;
using working_set_types = std::tuple<double>;

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
    const std::vector<float_t> y = { 1, 1, 1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1 };
    // const std::vector<std::uint32_t> ws_indices = { 13, 15, 5, 0, 4, 8, 6, 3, 9, 11, 12, 14, 10, 2, 7, 1 };
    // const std::vector<std::uint32_t> ws_indices = { 1, 2, 7, 9, 10, 11, 12, 14, 0, 3, 4, 5, 6, 8, 13, 15 };
    const std::vector<std::uint32_t> ws_indices = { 0, 1, 2, 7, 9, 10, 11, 12, 3, 4, 5, 6, 8, 13, 15, 14 };

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