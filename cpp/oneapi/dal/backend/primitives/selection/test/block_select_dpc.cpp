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
#include "oneapi/dal/backend/primitives/selection/block_select.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <typename TestType>
class block_select_test : public te::policy_fixture {
public:
    using Float = TestType;

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<Float>();
    }


    void test_selection(ndarray<Float, 2>& data, std::int64_t n, std::int64_t m, std::int64_t k) {
        SECTION("Output of selected values") {
            auto value_array = ndarray<Float, 2>::empty(get_queue(), { n, k });
            ndarray<int, 2> dummy_array;
            block_select<Float, true, false>(get_queue(), data, k, value_array, dummy_array).wait_and_throw();
            check_results<true, false>(data, value_array, dummy_array);

        }

        SECTION("Output of selected indices") {
            auto index_array = ndarray<int, 2>::empty(get_queue(), { n, k });
            ndarray<Float, 2> dummy_array;
            block_select<Float, false, true>(get_queue(), data, k, dummy_array, index_array).wait_and_throw();
            check_results<false, true>(data, dummy_array, index_array);
        }

        SECTION("Output of both") {
            auto value_array = ndarray<Float, 2>::empty(get_queue(), { n, k });
            auto index_array = ndarray<int, 2>::empty(get_queue(), { n, k });
            block_select<Float, true, true>(get_queue(), data, k, value_array, index_array).wait_and_throw();
            check_results<true, true>(data, value_array, index_array);
        }
    }

    template <bool selected_out, bool indices_out>
    void check_results(const ndarray<Float, 2>& block, const ndarray<Float, 2>& selected, 
                       const ndarray<int, 2>& indices) {
        ONEDAL_ASSERT(block.get_dimension(1) == selected.get_dimension(1));
        ONEDAL_ASSERT(block.get_dimension(1) == indices.get_dimension(1));

        auto k = selected.get_dimension(0);
        auto row_size = block.get_dimension(0);
        auto row_count = block.get_dimension(1);

        for(std::int64_t i = 0; i < row_count; i++) {
            auto max_val = std::numeric_limits<Float>::min();
            for(std::int64_t j = 0; j < k; j++) {
                Float cur_val = get_value<selected_out, indices_out>(block, selected, indices, k, row_size, i, j);
                check_presence_in_data<selected_out, indices_out>(block, selected, indices, k, row_size, i, j, cur_val);
                if(max_val < cur_val) max_val = cur_val;
            }
            for(std::int64_t j = 0; j < row_size; j++) {
                Float cur_val = block.get_data()[i * row_size + j];
                if(cur_val < max_val) {
                    check_presence_in_selection<selected_out, indices_out>(selected, indices, k, row_size, i, j, cur_val);
                }
            }
        }
    }

    template <bool selected_out, bool indices_out>
    Float get_value(const ndarray<Float, 2>& block, const ndarray<Float, 2>& selected, const ndarray<int, 2>& indices, 
                    std::int64_t k, std::int64_t row_size, std::int64_t row, std::int64_t pos) {
        if constexpr (selected_out) {
            return selected.get_data()[row * k + pos];
        }
        if constexpr (indices_out) {
            auto cur_index = indices.get_data()[row * k + pos];
            REQUIRE(cur_index > -1);
            REQUIRE(cur_index < row_size);
            return block.get_data()[row * row_size + cur_index];
        }
    }

    template <bool selected_out, bool indices_out>
    void check_presence_in_data(const ndarray<Float, 2>& block, const ndarray<Float, 2>& selected, const ndarray<int, 2>& indices, 
                    std::int64_t k,  std::int64_t row_size, std::int64_t row, std::int64_t pos, Float cur_val) {
        if constexpr (indices_out && selected_out) {
            REQUIRE(selected.get_data()[row * k + pos] == 
                    block.get_data()[row * row_size +indices.get_data()[row * k + pos]]);
        }
        if constexpr (!indices_out) {
            std::int64_t count = 0;
            for(std::int64_t l = 0; l < row_size; l++) {
                count += (std::int64_t) (block.get_data()[row * row_size + l] == cur_val);
            }
            REQUIRE(count > 0);
        }
    }

    template <bool selected_out, bool indices_out>
    void check_presence_in_selection(const ndarray<Float, 2>& selected, const ndarray<int, 2>& indices,
                                    std::int64_t k, std::int64_t row_size, std::int64_t row, std::int64_t pos, Float cur_val) {
        std::int64_t count = 0;
        if constexpr (!indices_out) {
            for(std::int64_t l = 0; l < k; l++) {
                count += (std::int64_t) (selected.get_data()[row * row_size + l] == cur_val);
            }
        } else {
            for(std::int64_t l = 0; l < k; l++) {
                count += (std::int64_t) (indices.get_data()[row * row_size + l] == pos);
            }
        }
        REQUIRE(count > 0);
    }
};

using selection_types =  std::tuple<float, double>;

TEMPLATE_LIST_TEST_M(block_select_test,
                     "selection degenerated test (k == 1)",
                     "[block select][small]",
                     selection_types) {
    using Float = TestType;
    Float data[] = { 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 4.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 5.0f, 1.0f };
    auto data_array = ndarray<Float, 2>::wrap(data, { 3, 5 });
    this->test_selection(data_array, 3, 5, 1);
}

TEMPLATE_LIST_TEST_M(block_select_test,
                     "selection single row test (k == 2)",
                     "[block select][small]",
                     selection_types) {
    using Float = TestType;
    Float data[] = { 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 4.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 5.0f, 1.0f };
    auto data_array = ndarray<Float, 2>::wrap(data, { 1, 15 });
    this->test_selection(data_array, 1, 15, 2);
}

TEMPLATE_LIST_TEST_M(block_select_test,
                     "selection degenerated test (k == m)",
                     "[block select][small]",
                     selection_types) {
    using Float = TestType;
    Float data[] = { 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 4.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 5.0f, 1.0f };
    auto data_array = ndarray<Float, 2>::wrap(data, { 3, 5 });
    this->test_selection(data_array, 3, 5, 3);
}

TEMPLATE_LIST_TEST_M(block_select_test,
                     "selection test",
                     "[block select][small]",
                     selection_types) {
    using Float = TestType;
    Float data[] = { 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 4.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 5.0f, 1.0f };
    auto data_array = ndarray<Float, 2>::wrap(data, { 3, 5 });
    this->test_selection(data_array, 3, 5, 2);
}

TEMPLATE_LIST_TEST_M(block_select_test,
                     "selection test on random data",
                     "[block select][medium]",
                     selection_types) {
    using Float = TestType;
    const auto df = GENERATE_DATAFRAME(
        te::dataframe_builder{ 17, 33 }.fill_uniform(-0.2, 0.5));
    const table df_table = df.get_table(this->get_homogen_table_id());
    const auto df_rows = row_accessor<const Float>(df_table).pull({ 0, -1 });
    auto data_array = ndarray<Float, 2>::wrap(df_rows.get_data(), { 3, 5 });
    this->test_selection(data_array, 3, 5, 2);
}

} // namespace oneapi::dal::backend::primitives::test
