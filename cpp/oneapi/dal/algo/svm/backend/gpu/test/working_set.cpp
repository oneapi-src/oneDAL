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

#include "oneapi/dal/algo/svm/backend/gpu/working_set.hpp"

namespace oneapi::dal::svm::backend::test {

namespace pr = dal::backend::primitives;
namespace te = dal::test::engine;
namespace la = te::linalg;

template <typename TestType>
class working_set_test : public te::policy_fixture {
public:
    using Float = TestType;

    void test_working_set(const std::vector<Float>& f,
                          const std::vector<Float>& y,
                          const std::vector<Float>& alpha,
                          const Float C,
                          const std::int64_t n_vectors,
                          const std::int64_t expected_n_ws,
                          const std::vector<std::uint32_t>& expected_ws_indices) {
        auto& q = this->get_queue();
        INFO("Allocate ndarray");
        auto f_ndarray = pr::ndarray<Float, 1>::empty(q, { n_vectors });
        auto y_ndarray = pr::ndarray<Float, 1>::empty(q, { n_vectors });
        auto alpha_ndarray = pr::ndarray<Float, 1>::empty(q, { n_vectors });
        dal::backend::copy<Float>(q, f_ndarray.get_mutable_data(), f.data(), n_vectors)
            .wait_and_throw();
        dal::backend::copy<Float>(q, y_ndarray.get_mutable_data(), y.data(), n_vectors)
            .wait_and_throw();
        dal::backend::copy<Float>(q, alpha_ndarray.get_mutable_data(), alpha.data(), n_vectors)
            .wait_and_throw();

        INFO("Init working set");
        auto ws = working_set<Float>(q);
        ws.init(n_vectors);

        INFO("Check n_ws");
        REQUIRE(ws.get_size() == expected_n_ws);

        INFO("Run select_ws");
        ws.select_ws(y_ndarray, alpha_ndarray, f_ndarray, C).wait_and_throw();

        INFO("Check ws_indices");
        const auto indices = ws.get_ws_indices();
        const auto indices_arr = indices.flatten(q);

        const auto indices_mat_host = la::matrix<std::uint32_t>::wrap(indices_arr).to_host();
        const auto indices_arr_host = indices_mat_host.get_array();
        for (std::int64_t i = 0; i < indices.get_dimension(0); i++)
            REQUIRE(indices_arr_host[i] == expected_ws_indices[i]);
    }
};

using working_set_types = std::tuple<float, double>;

TEMPLATE_LIST_TEST_M(working_set_test,
                     "select ws common flow",
                     "[svm][working_set]",
                     working_set_types) {
    using float_t = TestType;

    constexpr std::int64_t n_vectors = 9;

    const std::vector<float_t> f = { 2, 7, 3, 9, 5, 1, 8, 4, 6 };
    const std::vector<float_t> y = { -1, -1, 1, 1, -1, -1, 1, -1, 1 };
    const std::vector<float_t> alpha = { 1.5, 0, 5.0, 1.2, 0.2, 0, 5, 4.3, 0 };

    constexpr float_t C = 5.0;

    constexpr std::int64_t expected_n_ws = 8;

    const std::vector<std::uint32_t> expected_ws_indices = { 0, 7, 4, 8, 2, 1, 6, 3 };

    this->test_working_set(f, y, alpha, C, n_vectors, expected_n_ws, expected_ws_indices);
}

TEMPLATE_LIST_TEST_M(working_set_test,
                     "not enough elements in upper set",
                     "[svm][working_set]",
                     working_set_types) {
    using float_t = TestType;

    constexpr std::int64_t n_vectors = 10;

    const std::vector<float_t> f = { 10, 2, 3, 6, 9, 1, 7, 5, 4, 8 };
    const std::vector<float_t> y = { -1, -1, -1, -1, -1, 1, 1, 1, 1, 1 };
    const std::vector<float_t> alpha = { 0.0, 0.0, 0.0, 1.3, 0.0, 1.5, 2.0, 2.5, 2.5, 2.5 };

    constexpr float_t C = 2.5;

    constexpr std::int64_t expected_n_ws = 8;

    const std::vector<std::uint32_t> expected_ws_indices = { 5, 3, 6, 8, 7, 9, 4, 0 };

    this->test_working_set(f, y, alpha, C, n_vectors, expected_n_ws, expected_ws_indices);
}

TEMPLATE_LIST_TEST_M(working_set_test,
                     "not enough elements in lower set",
                     "[svm][working_set]",
                     working_set_types) {
    using float_t = TestType;

    constexpr std::int64_t n_vectors = 10;

    const std::vector<float_t> f = { 10, 2, 3, 6, 9, 1, 7, 5, 4, 8 };
    const std::vector<float_t> y = { -1, -1, -1, -1, -1, 1, 1, 1, 1, 1 };
    const std::vector<float_t> alpha = { 0.0, 2.5, 2.5, 2.5, 1.5, 0.0, 0.0, 0.0, 2.5, 0.0 };

    constexpr float_t C = 2.5;

    constexpr std::int64_t expected_n_ws = 8;

    const std::vector<std::uint32_t> expected_ws_indices = { 5, 1, 2, 7, 8, 4, 0, 3 };

    this->test_working_set(f, y, alpha, C, n_vectors, expected_n_ws, expected_ws_indices);
}

} // namespace oneapi::dal::svm::backend::test
