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

#include <array>

#include "oneapi/dal/algo/jaccard/vertex_similarity.hpp"

#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::algo::jaccard::test {

TEST("neigh_u[0] > neigh_v[n_v - 1] && n_u >= 16 && n_v >= 16") {
    std::array<std::int32_t, 18> neigh_u = { 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                             25, 26, 27, 28, 29, 30, 31, 32, 33 };
    std::array<std::int32_t, 16> neigh_v = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    REQUIRE(oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(),
                                                                neigh_v.data(),
                                                                18,
                                                                16) == 0);
}

TEST("neigh_v[0] > neigh_u[n_u - 1] && n_u >= 16 && n_v >= 16") {
    std::array<std::int32_t, 16> neigh_u = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    std::array<std::int32_t, 18> neigh_v = { 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                             25, 26, 27, 28, 29, 30, 31, 32, 33 };

    REQUIRE(oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(),
                                                                neigh_v.data(),
                                                                16,
                                                                18) == 0);
}

TEST("neigh_u[0] > neigh_v[n_v - 1] && n_u >= 8 && n_v >= 8") {
    std::array<std::int32_t, 11> neigh_u = { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    std::array<std::int32_t, 10> neigh_v = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    REQUIRE(oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(),
                                                                neigh_v.data(),
                                                                11,
                                                                10) == 0);
}

TEST("neigh_v[0] > neigh_u[n_u - 1] && n_u >= 8 && n_v >= 8") {
    std::array<std::int32_t, 10> neigh_u = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::array<std::int32_t, 11> neigh_v = { 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    REQUIRE(oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(),
                                                                neigh_v.data(),
                                                                10,
                                                                11) == 0);
}

TEST("neigh_u[0] > neigh_v[n_v - 1] && 4 <= n_u < 8 && 4 <= n_v < 8") {
    std::array<std::int32_t, 5> neigh_u = { 4, 5, 6, 7, 8 };
    std::array<std::int32_t, 4> neigh_v = { 0, 1, 2, 3 };
    REQUIRE(+oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(),
                                                                 neigh_v.data(),
                                                                 5,
                                                                 4) == 0);
}

TEST("neigh_v[0] > neigh_u[n_u - 1] && 4 <= n_u < 8 && 4 <= n_v < 8") {
    std::array<std::int32_t, 4> neigh_u = { 0, 1, 2, 3 };
    std::array<std::int32_t, 5> neigh_v = { 4, 5, 6, 7, 8 };
    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 4, 5) ==
        0);
}

TEST("neigh_u[0] > neigh_v[n_v - 1] && n_u < 4 && n_v < 4") {
    std::array<std::int32_t, 2> neigh_u = { 3, 4 };
    std::array<std::int32_t, 3> neigh_v = { 0, 1, 2 };
    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 2, 3) ==
        0);
}

TEST("neigh_v[0] > neigh_u[n_u - 1] && n_u < 4 && n_v < 4") {
    std::array<std::int32_t, 3> neigh_u = { 0, 1, 2 };
    std::array<std::int32_t, 2> neigh_v = { 3, 4 };
    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 3, 2) ==
        0);
}

TEST("min_neigh_u > max_neigh_v && min_neigh_u < neigh_v[n_v - 1] && n_u >= 16 && n_v >= 16") {
    std::array<std::int32_t, 16> neigh_u = { 16, 17, 18, 19, 20, 21, 22, 23,
                                             24, 25, 26, 27, 28, 29, 30, 31 };
    std::array<std::int32_t, 20> neigh_v = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                                             10, 11, 12, 13, 14, 15, 19, 21, 27, 30 };

    REQUIRE(oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(),
                                                                neigh_v.data(),
                                                                16,
                                                                20) == 4);
}

TEST("min_neigh_v > max_neigh_u && min_neigh_v < neigh_u[n_u - 1] && n_u >= 16 && n_v >= 16") {
    std::array<std::int32_t, 20> neigh_u = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                                             10, 11, 12, 13, 14, 15, 19, 21, 27, 30 };
    std::array<std::int32_t, 16> neigh_v = { 16, 17, 18, 19, 20, 21, 22, 23,
                                             24, 25, 26, 27, 28, 29, 30, 31 };

    REQUIRE(oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(),
                                                                neigh_v.data(),
                                                                20,
                                                                16) == 4);
}

TEST("min_neigh_u > max_neigh_v && min_neigh_u < neigh_v[n_v - 1] && n_u >= 8 && n_v >= 8") {
    std::array<std::int32_t, 8> neigh_u = { 12, 13, 14, 15, 16, 17, 18, 19 };
    std::array<std::int32_t, 12> neigh_v = { 0, 1, 2, 3, 4, 5, 6, 7, 13, 15, 19, 21 };
    REQUIRE(oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(),
                                                                neigh_v.data(),
                                                                8,
                                                                12) == 3);
}

TEST("min_neigh_v > max_neigh_u && min_neigh_v < neigh_u[n_u - 1] && n_u >= 8 && n_v >= 8") {
    std::array<std::int32_t, 12> neigh_u = { 0, 1, 2, 3, 4, 5, 6, 7, 13, 15, 19, 21 };
    std::array<std::int32_t, 8> neigh_v = { 12, 13, 14, 15, 16, 17, 18, 19 };

    REQUIRE(oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(),
                                                                neigh_v.data(),
                                                                12,
                                                                8) == 3);
}

TEST(
    "min_neigh_u > max_neigh_v && min_neigh_u < neigh_v[n_v - 1] && 4 <= n_u < 8 && 4 <= n_v < 8 ") {
    std::array<std::int32_t, 4> neigh_u = { 13, 14, 15, 16 };
    std::array<std::int32_t, 5> neigh_v = { 0, 1, 2, 3, 17 };
    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 4, 5) ==
        0);
}

TEST(
    "min_neigh_v > max_neigh_u && min_neigh_v < neigh_u[n_u - 1] && 4 <= n_u < 8 && 4 <= n_v < 8") {
    std::array<std::int32_t, 5> neigh_u = { 0, 1, 2, 3, 17 };
    std::array<std::int32_t, 4> neigh_v = { 13, 14, 15, 16 };

    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 5, 4) ==
        0);
}

TEST("Several intersections for blocks of 16 elements") {
    std::array<std::int32_t, 16> neigh_u = {
        3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
    };
    std::array<std::int32_t, 16> neigh_v = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    REQUIRE(oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(),
                                                                neigh_v.data(),
                                                                16,
                                                                16) == 13);
}

TEST("Several intersections for blocks of 8 elements") {
    std::array<std::int32_t, 8> neigh_u = { 3, 4, 5, 6, 7, 8, 9, 10 };
    std::array<std::int32_t, 8> neigh_v = { 0, 1, 2, 3, 4, 5, 6, 7 };

    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 8, 8) ==
        5);
}

TEST("Several intersections for blocks of 4 elements") {
    std::array<std::int32_t, 4> neigh_u = { 2, 3, 4, 5 };
    std::array<std::int32_t, 4> neigh_v = { 0, 1, 2, 3 };

    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 4, 4) ==
        2);
}

TEST("Several intersections for < 4 elements") {
    std::array<std::int32_t, 2> neigh_u = { 3, 4 };
    std::array<std::int32_t, 3> neigh_v = { 0, 4, 5 };
    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 2, 3) ==
        1);
}

TEST("Half-vectorized branch for block of 16 elements") {
    std::array<std::int32_t, 5> neigh_u = { 2, 4, 6, 8, 10 };
    std::array<std::int32_t, 16> neigh_v = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    REQUIRE(oneapi::dal::preview::jaccard::detail::intersection(neigh_v.data(),
                                                                neigh_u.data(),
                                                                16,
                                                                5) == 5);
    REQUIRE(oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(),
                                                                neigh_v.data(),
                                                                5,
                                                                16) == 5);
}

TEST("Half-vectorized branch for block of 8 elements") {
    std::array<std::int32_t, 5> neigh_u = { 2, 4, 6, 8, 10 };
    std::array<std::int32_t, 8> neigh_v = { 0, 1, 2, 3, 4, 5, 6, 7 };

    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 5, 8) ==
        3);

    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_v.data(), neigh_u.data(), 8, 5) ==
        3);
}

TEST("Half-vectorized branch for block of 4 elements") {
    std::array<std::int32_t, 2> neigh_u = { 1, 3 };
    std::array<std::int32_t, 5> neigh_v = { 0, 1, 2, 3, 4 };

    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 2, 5) ==
        2);
    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_v.data(), neigh_u.data(), 5, 2) ==
        2);
}

TEST("Equal blocks of 16 elements") {
    std::array<std::int32_t, 16> neigh_u = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    std::array<std::int32_t, 16> neigh_v = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    REQUIRE(oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(),
                                                                neigh_v.data(),
                                                                16,
                                                                16) == 16);
}

TEST("Equal blocks of 8 elements") {
    std::array<std::int32_t, 8> neigh_u = { 0, 1, 2, 3, 4, 5, 6, 7 };
    std::array<std::int32_t, 8> neigh_v = { 0, 1, 2, 3, 4, 5, 6, 7 };
    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 8, 8) ==
        8);
}

TEST("Equal blocks of 4 elements") {
    std::array<std::int32_t, 4> neigh_u = { 0, 1, 2, 3 };
    std::array<std::int32_t, 4> neigh_v = { 0, 1, 2, 3 };
    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 4, 4) ==
        4);
}

TEST("Isolated vertex") {
    std::array<std::int32_t, 8> neigh_u = { 0, 1, 2, 3, 4, 5, 6, 7 };
    std::array<std::int32_t, 5> neigh_v = { 2, 4, 6, 8, 10 };
    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 5, 0) ==
        0);
    REQUIRE(
        oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(), neigh_v.data(), 0, 5) ==
        0);
}

TEST("Isolated vertex && n = 16") {
    std::array<std::int32_t, 16> neigh_u = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    std::array<std::int32_t, 8> neigh_v = { 0, 1, 2, 3, 4, 5, 6, 7 };

    REQUIRE(oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(),
                                                                neigh_v.data(),
                                                                16,
                                                                0) == 0);
    REQUIRE(oneapi::dal::preview::jaccard::detail::intersection(neigh_u.data(),
                                                                neigh_v.data(),
                                                                0,
                                                                16) == 0);
}

} // namespace oneapi::dal::algo::jaccard::test
