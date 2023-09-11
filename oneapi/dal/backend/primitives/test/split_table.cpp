/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/split_table.hpp"

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/table/detail/table_builder.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

class c_order {};
class f_order {};

template <typename order>
struct order_map {};

template <>
struct order_map<c_order> {
    constexpr static auto value = ndorder::c;
};

template <>
struct order_map<f_order> {
    constexpr static auto value = ndorder::f;
};

template <typename order>
constexpr auto order_v = order_map<order>::value;

template <typename T, ndorder order>
inline auto table_to_ndarray(const table& t) {
    const auto rc = t.get_row_count();
    const auto cc = t.get_column_count();
    auto res = ndarray<T, 2, order>::empty({ rc, cc });
    for (std::int64_t r = 0; r < rc; ++r) {
        const auto row = row_accessor<const T>(t).pull({ r, r + 1 });
        for (std::int64_t c = 0; c < cc; ++c) {
            res.at(r, c) = row[c];
        }
    }
    return res;
}

template <typename TestType>
class split_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
    using float_t = std::tuple_element_t<0, TestType>;
    using order_t = std::tuple_element_t<1, TestType>;
    static constexpr auto order = order_v<order_t>;

public:
    void generate() {
        m_ = GENERATE(2, 11, 17, 32, 37);
        d_ = GENERATE(3, 10, 17, 32, 37);
        b_ = GENERATE(2, 28, 43);
        generate_data();
    }

    void generate_data() {
        const auto input_df =
            GENERATE_DATAFRAME(te::dataframe_builder{ m_, d_ }.fill_uniform(-0.2, 0.5));
        this->input_ = input_df.get_table(this->get_policy(), this->get_homogen_table_id());
    }

    bool is_initialized() const {
        return m_ > 0 && b_ > 0 && d_ > 0;
    }

    void check_if_initialized() {
        if (!is_initialized()) {
            throw std::runtime_error{ "test is not initialized" };
        }
    }

    auto get_input_view() {
        return table_to_ndarray<float_t, order>(this->train_);
    }

#ifdef ONEDAL_DATA_PARALLEL
    void test_correctness() {
        check_if_initialized();

        auto& queue = this->get_queue();
        auto view = table_to_ndarray<float_t, order>(this->input_);
        auto split = split_table<float_t>(queue, this->input_, this->b_);

        CAPTURE(m_, b_, d_);
        const std::int64_t b_count = split.size();
        uniform_blocking blocking{ this->m_, this->b_ };
        REQUIRE(b_count == blocking.get_block_count());

        for (std::int64_t r = 0; r < this->m_; ++r) {
            const auto b = r / this->b_;
            const auto l = r % this->b_;

            // new block corresponds to l == 0
            const auto slice = split.at(b).to_host(queue);

            for (std::int64_t c = 0; c < this->d_; ++c) {
                const auto gtr = view.at(r, c);
                const auto val = slice.at(l, c);

                CAPTURE(r, c, b, l, gtr, val);
                REQUIRE(gtr == val);
            }
        }
    }
#else // ONEDAL_DATA_PARALLEL
    void test_correctness() {
        check_if_initialized();

        auto split = split_table<float_t>(this->input_, this->b_);
        auto view = table_to_ndarray<float_t, order>(this->input_);

        CAPTURE(m_, b_, d_);
        const std::int64_t b_count = split.size();
        uniform_blocking blocking{ this->m_, this->b_ };
        REQUIRE(b_count == blocking.get_block_count());

        for (std::int64_t r = 0; r < this->m_; ++r) {
            const auto b = r / this->b_;
            const auto l = r % this->b_;

            const auto slice = split.at(b);

            for (std::int64_t c = 0; c < this->d_; ++c) {
                const auto gtr = view.at(r, c);
                const auto val = slice.at(l, c);

                CAPTURE(r, c, b, l, gtr, val);
                REQUIRE(gtr == val);
            }
        }
    }
#endif // ONEDAL_DATA_PARALLEL

private:
    table input_;
    std::int64_t m_, d_, b_;
};

using split_types = COMBINE_TYPES((float, double), (c_order, f_order));

TEMPLATE_LIST_TEST_M(split_test, "Randomly filled split", "[split][small]", split_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate();
    this->test_correctness();
}

} // namespace oneapi::dal::backend::primitives::test
