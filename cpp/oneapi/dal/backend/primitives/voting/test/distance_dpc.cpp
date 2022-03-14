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
#include "oneapi/dal/backend/primitives/voting.hpp"

#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/math.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace de = dal::detail;
namespace la = te::linalg;

class quasi_uniform_test : public te::policy_fixture {
public:
    using cls_t = std::int32_t;
    using dst_t = float;

    auto compute_results(const ndview<cls_t, 2>& responses) {
        const auto r = responses.get_dimension(0);
        const auto k = responses.get_dimension(1);
        auto tmp = ndarray<cls_t, 2>::empty(this->get_queue(), { r, k });
        auto res = ndarray<cls_t, 1>::empty(this->get_queue(), { r });
        copy(this->get_queue(), tmp, responses).wait_and_throw();
        for (std::int32_t j = 0; j < r; ++j) {
            auto* const from = tmp.get_mutable_data() + j * k;
            auto* const to = tmp.get_mutable_data() + (j + 1) * k;
            std::sort(from, to);
            cls_t last = -1, winner = -1;
            std::int32_t last_span = -1, winner_span = -1;
            for (std::int32_t i = 0; i < k; ++i) {
                const cls_t& cur = *(from + i);
                if (cur == last) {
                    ++last_span;
                }
                else {
                    last = cur;
                    last_span = 0;
                }
                if (last_span > winner_span) {
                    winner = last;
                    winner_span = last_span;
                }
            }
            *(res.get_mutable_data() + j) = winner;
        }
        return res;
    }

    void test_correctness(const ndview<cls_t, 2>& responses, const ndview<cls_t, 1>& results) {
        const auto own_res = compute_results(responses);
        const auto r = results.get_dimension(0);
        for (std::int32_t i = 0; i < r; ++i) {
            const auto val = *(results.get_data() + i);
            const auto gtr = *(own_res.get_data() + i);
            CAPTURE(i, val, gtr);
            REQUIRE(val == gtr);
        }
    }

    auto generate_indices(std::int32_t m, std::int32_t n) {
        auto x = ndarray<cls_t, 2>::empty(this->get_queue(), { m, n });

        for (std::int32_t j = 0; j < m; ++j) {
            for (std::int32_t i = 0; i < n; ++i) {
                if (i < (j + 1)) {
                    x.at(j, i) = i;
                }
                else {
                    x.at(j, i) = i - 1;
                }
            }
        }

        return x;
    }

    auto generate_distances(std::int32_t m, std::int32_t n) {
        auto x = ndarray<dst_t, 2>::empty(this->get_queue(), { m, n });

        constexpr dst_t dst = 0.1;

        fill(this->get_queue(), x, dst).wait_and_throw();

        return x;
    }

    auto generate_input() {
        return std::make_tuple(generate_indices(m_, n_), generate_distances(m_, n_));
    }

    void generate() {
        this->m_ = GENERATE(1, 16, 128, 1024);
        this->n_ = GENERATE(1, 16, 128, 1024);
    }

    void test_pipeline() {
        auto [x, d] = this->generate_input();
        auto y = ndarray<cls_t, 1>::empty(this->get_queue(), { m_ });

        auto voting = make_distance_voting<dst_t, cls_t>(this->get_queue(), m_, n_);

        voting->operator()(x, d, y).wait_and_throw();

        test_correctness(x, y);
    }

private:
    std::int64_t m_, n_;
};

TEST_M(quasi_uniform_test, "quasi uniform voting - naive", "[ndarray]") {
    this->generate();
    this->test_pipeline();
}

class spec_dist_test : public te::policy_fixture {
public:
    using cls_t = std::int32_t;
    using dst_t = float;

    auto compute_results(const ndview<cls_t, 2>& responses) {
        const auto r = responses.get_dimension(0);
        const auto k = responses.get_dimension(1);
        auto res = ndarray<cls_t, 1>::empty(this->get_queue(), { r });
        for (std::int32_t i = 0; i < r; ++i) {
            res.at(i) = (i < k) ? (k - i - 1) : 0;
        }
        return res;
    }

    void test_correctness(const ndview<cls_t, 2>& responses, const ndview<cls_t, 1>& results) {
        const auto own_res = compute_results(responses);
        const auto r = results.get_dimension(0);
        for (std::int32_t i = 0; i < r; ++i) {
            const auto val = *(results.get_data() + i);
            const auto gtr = *(own_res.get_data() + i);
            CAPTURE(i, val, gtr);
            REQUIRE(val == gtr);
        }
    }

    auto generate_indices(std::int32_t m, std::int32_t n) {
        auto x = ndarray<cls_t, 2>::empty(this->get_queue(), { m, n });

        for (std::int32_t j = 0; j < m; ++j) {
            for (std::int32_t i = 0; i < n; ++i) {
                x.at(j, i) = i;
            }
        }

        return x;
    }

    auto generate_distances(std::int32_t m, std::int32_t n) {
        auto x = ndarray<dst_t, 2>::empty(this->get_queue(), { m, n });

        for (std::int32_t j = 0; j < m; ++j) {
            for (std::int32_t i = 0; i < n; ++i) {
                if (j == (n - i - 1)) {
                    x.at(j, i) = 0.1;
                }
                else {
                    x.at(j, i) = 0.2;
                }
            }
        }

        return x;
    }

    auto generate_input() {
        return std::make_tuple(generate_indices(m_, n_), generate_distances(m_, n_));
    }

    void generate() {
        this->m_ = GENERATE(1, 16, 128, 1024);
        this->n_ = GENERATE(1, 16, 128, 1024);
    }

    void test_pipeline() {
        auto [x, d] = this->generate_input();
        auto y = ndarray<cls_t, 1>::empty(this->get_queue(), { m_ });

        auto voting = make_distance_voting<dst_t, cls_t>(this->get_queue(), m_, n_);

        voting->operator()(x, d, y).wait_and_throw();

        test_correctness(x, y);
    }

private:
    std::int64_t m_, n_;
};

TEST_M(spec_dist_test, "specific element distance - naive", "[ndarray]") {
    this->generate();
    this->test_pipeline();
}

} // namespace oneapi::dal::backend::primitives::test
