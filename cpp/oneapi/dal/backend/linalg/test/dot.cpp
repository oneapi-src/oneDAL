/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <iostream>

#include "oneapi/dal/test/common.hpp"
#include "oneapi/dal/backend/linalg/dot.hpp"

namespace oneapi::dal::backend::linalg::test {

void check_ones_matrix_dot(std::int64_t m, std::int64_t n, std::int64_t k, const matrix<float>& c) {
    REQUIRE(c.get_shape() == shape{ m, n });
    c.for_each([&](float x) {
        REQUIRE(std::int64_t(x) == k);
    });
}

TEST_CASE("matrix dot simple", "[linalg][host]") {
    const std::int64_t m = GENERATE(3, 6);
    const std::int64_t n = GENERATE(4, 7);
    const std::int64_t k = GENERATE(5, 8);
    const layout l = GENERATE(layout::row_major, layout::column_major);

    const auto stringify_params = [&]() {
        return fmt::format("m = {}, n = {}, k = {}, l = {}", m, n, k, l);
    };

    SECTION("A x B, " + stringify_params()) {
        const auto A = matrix<float>::ones({ m, k }, l);
        const auto B = matrix<float>::ones({ k, n }, l);
        const auto C = dot(A, B);
        check_ones_matrix_dot(m, n, k, C);
    }

    SECTION("A^T x B, " + stringify_params()) {
        const auto A = matrix<float>::ones({ k, m }, l);
        const auto B = matrix<float>::ones({ k, n }, l);
        const auto C = dot(A.T(), B);
        check_ones_matrix_dot(m, n, k, C);
    }

    SECTION("A x B^T, " + stringify_params()) {
        const auto A = matrix<float>::ones({ m, k }, l);
        const auto B = matrix<float>::ones({ n, k }, l);
        const auto C = dot(A, B.T());
        check_ones_matrix_dot(m, n, k, C);
    }

    SECTION("A^T x B^T, " + stringify_params()) {
        const auto A = matrix<float>::ones({ k, m }, l);
        const auto B = matrix<float>::ones({ n, k }, l);
        const auto C = dot(A.T(), B.T());
        check_ones_matrix_dot(m, n, k, C);
    }
}

TEST_CASE("matrix dot in-place", "[linalg][host]") {
    const std::int64_t m = GENERATE(3, 6);
    const std::int64_t n = GENERATE(4, 7);
    const std::int64_t k = GENERATE(5, 8);
    const layout a_l = GENERATE(layout::row_major, layout::column_major);
    const layout b_l = GENERATE(layout::row_major, layout::column_major);
    const layout c_l = GENERATE(layout::row_major, layout::column_major);

    const auto stringify_params = [&]() {
        return fmt::format("m = {}, n = {}, k = {}, a_l = {}, b_l = {}, c_l = {}",
                           m,
                           n,
                           k,
                           a_l,
                           b_l,
                           c_l);
    };

    SECTION("A x B, " + stringify_params()) {
        const auto A = matrix<float>::ones({ m, k }, a_l);
        const auto B = matrix<float>::ones({ k, n }, b_l);
        auto C = matrix<float>::empty({ m, n }, c_l);
        dot(A, B, C);
        check_ones_matrix_dot(m, n, k, C);
    }

    SECTION("A^T x B, " + stringify_params()) {
        const auto A = matrix<float>::ones({ k, m }, a_l);
        const auto B = matrix<float>::ones({ k, n }, b_l);
        auto C = matrix<float>::empty({ m, n }, c_l);
        dot(A.T(), B, C);
        check_ones_matrix_dot(m, n, k, C);
    }

    SECTION("A x B^T, " + stringify_params()) {
        const auto A = matrix<float>::ones({ m, k }, a_l);
        const auto B = matrix<float>::ones({ n, k }, b_l);
        auto C = matrix<float>::empty({ m, n }, c_l);
        dot(A, B.T(), C);
        check_ones_matrix_dot(m, n, k, C);
    }

    SECTION("A^T x B^T, " + stringify_params()) {
        const auto A = matrix<float>::ones({ k, m }, a_l);
        const auto B = matrix<float>::ones({ n, k }, b_l);
        auto C = matrix<float>::empty({ m, n }, c_l);
        dot(A.T(), B.T(), C);
        check_ones_matrix_dot(m, n, k, C);
    }
}

} // namespace oneapi::dal::backend::linalg::test
