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

#include <type_traits>
#include <tuple>

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/selection/select_indexed.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

template <typename TestType>
class selection_by_rows_test : public te::float_algo_fixture<TestType> {
public:
    using float_t = TestType;

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<float_t>();
    }

    void generate() {
        m_ = GENERATE(1, 11, 129);
        n_ = GENERATE(1, 17, 131);
        k_ = GENERATE(1, 19, 137);
    }

    void test_selection() {
        auto [dst, dst_event] = destination();
        auto ids = indices();
        auto src = source();

        select_indexed(this->get_queue(), ids, src, dst, { dst_event }).wait_and_throw();

        check_selection(ids, src, dst);
    }

    void check_selection(   const ndview<std::int32_t, 2>& ids,
                            const ndview<TestType, 2>& src,
                            const ndview<TestType, 2>& dst) {
        for(std::int64_t j = 0; j < m_; ++j) {
            for(std::int64_t i = 0; i < n_; ++i) {
                const std::int64_t idx = (i + j) % k_;
                const std::int64_t val = j + idx;
                CAPTURE(i, j, idx, val);
                //CAPTURE(ids.at(i, j), src.at(idx, j), dst.at(i, j));
                REQUIRE(TestType(val) == dst.at(j, i));
            }
        }
    }

    auto destination() {
        return ndarray<TestType, 2>::zeros(this->get_queue(), { m_, n_ });
    }

    auto indices() {
        auto res = ndarray<std::int32_t, 2>::empty(this->get_queue(), { m_, n_ });
        for(std::int64_t j = 0; j < m_; ++j) {
            for(std::int64_t i = 0; i < n_; ++i) {
                res.at(j, i) = (i + j) % k_;
            }
        }
        return res;
    }

    auto source() {
        auto res = ndarray<TestType, 2>::empty(this->get_queue(), { m_, k_ });
        for(std::int64_t j = 0; j < m_; ++j) {
            for(std::int64_t i = 0; i < k_; ++i) {
                res.at(j, i) = i + j;
            }
        }
        return res;
    }

private:
    std::int64_t m_, n_, k_;
};

using selection_types = std::tuple<float, double>;

TEMPLATE_LIST_TEST_M(selection_by_rows_test,
                     "selection indexed",
                     "[block select][small]",
                     selection_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate();
    this->test_selection();
}

} // namespace oneapi::dal::backend::primitives::test
