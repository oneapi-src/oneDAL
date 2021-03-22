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
#include "oneapi/dal/backend/primitives/selection/select_by_rows.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <typename TestType>
class selection_by_rows_test : public te::policy_fixture {
public:
    using Float = TestType;

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<Float>();
    }

    void test_selection(ndview<Float, 2>& data,
                        std::int64_t row_count,
                        std::int64_t col_count,
                        std::int64_t k) {
/*        INFO("Output of selected values") {
            selection_by_rows<Float> sel(data);
            ndarray<int, 2> dummy_array;
            auto value_array = ndarray<Float, 2>::empty(get_queue(), { row_count, k });
            sel.select(get_queue(), k, value_array).wait_and_throw();
            check_results<true, false>(data, value_array, dummy_array);
        }
        INFO("Output of selected indices") {
            selection_by_rows<Float> sel(data);
            ndarray<Float, 2> dummy_array;
            auto index_array = ndarray<int, 2>::empty(get_queue(), { row_count, k });
            sel.select(get_queue(), k, index_array).wait_and_throw();
            check_results<false, true>(data, dummy_array, index_array);
        }
*/
        INFO("Output of both") {
            auto data_ptr = data.get_data();
            std::cout << "Data: ";
            for(int i = 0; i < col_count; i++)
                std::cout << "data: " << i << " " << data_ptr[i] << std::endl;;
            std::cout << std::endl;
            selection_by_rows<Float> sel(data);
            auto value_array = ndarray<Float, 2>::empty(get_queue(), { row_count, k });
            auto index_array = ndarray<int, 2>::empty(get_queue(), { row_count, k });
            sel.select(get_queue(), k, value_array, index_array).wait_and_throw();
            check_results<true, true>(data, value_array, index_array);
        }
    }

    template <bool selection_out, bool indices_out>
    void check_results(const ndview<Float, 2>& data,
                       const ndview<Float, 2>& selection,
                       const ndview<int, 2>& indices) {
        ONEDAL_ASSERT(data.get_dimension(0) == selection.get_dimension(0));
        ONEDAL_ASSERT(data.get_dimension(0) == indices.get_dimension(0));

        auto k = selection.get_dimension(1);
        auto row_size = data.get_dimension(1);
        auto row_count = data.get_dimension(0);
//        auto data_ptr = data.get_data();
        auto selection_ptr = selection.get_data();
        auto indices_ptr = indices.get_data();
        for(int i = 0; i < row_size; i++)
            std::cout << "res: " << i << " " <<  selection_ptr[i] << " " << indices_ptr[i] << std::endl;

        for (std::int64_t i = 0; i < row_count; i++) {
            auto max_val = std::numeric_limits<Float>::lowest();
            for (std::int64_t j = 0; j < k; j++) {
                Float cur_val = get_value<selection_out, indices_out>(data,
                                                                      selection,
                                                                      indices,
                                                                      k,
                                                                      row_size,
                                                                      i,
                                                                      j);
                check_presence_in_data<selection_out, indices_out>(data,
                                                                   selection,
                                                                   indices,
                                                                   k,
                                                                   row_size,
                                                                   i,
                                                                   j,
                                                                   cur_val);
                if (max_val < cur_val)
                    max_val = cur_val;
            }
            for (std::int64_t j = 0; j < row_size; j++) {
                Float cur_val = data.get_data()[i * row_size + j];
                if (cur_val < max_val) {
                    check_presence_in_selection<selection_out, indices_out>(selection,
                                                                            indices,
                                                                            k,
                                                                            row_size,
                                                                            i,
                                                                            j,
                                                                            cur_val);
                }
            }
        }
    }

    template <bool selection_out, bool indices_out>
    Float get_value(const ndview<Float, 2>& data,
                    const ndview<Float, 2>& selection,
                    const ndview<int, 2>& indices,
                    std::int64_t k,
                    std::int64_t row_size,
                    std::int64_t row,
                    std::int64_t pos) {
        if constexpr (selection_out) {
            return selection.get_data()[row * k + pos];
        }
        if constexpr (indices_out) {
            auto cur_index = indices.get_data()[row * k + pos];
            REQUIRE(cur_index > -1);
            REQUIRE(cur_index < row_size);
            return data.get_data()[row * row_size + cur_index];
        }
    }

    template <bool selection_out, bool indices_out>
    void check_presence_in_data(const ndview<Float, 2>& data,
                                const ndview<Float, 2>& selection,
                                const ndview<int, 2>& indices,
                                std::int64_t k,
                                std::int64_t row_size,
                                std::int64_t row,
                                std::int64_t pos,
                                Float cur_val) {
        CAPTURE(row, k, pos);
        if constexpr (indices_out && selection_out) {
            REQUIRE(selection.get_data()[row * k + pos] ==
                    data.get_data()[row * row_size + indices.get_data()[row * k + pos]]);
        }
        std::int64_t count = 0;
        for (std::int64_t i = 0; i < row_size; i++) {
            Float probe = data.get_data()[row * row_size + i];
            count += (std::int64_t)(probe == cur_val);
        }
        REQUIRE(count > 0);
    }

    template <bool selection_out, bool indices_out>
    void check_presence_in_selection(const ndview<Float, 2>& selection,
                                     const ndview<int, 2>& indices,
                                     std::int64_t k,
                                     std::int64_t row_size,
                                     std::int64_t row,
                                     std::int64_t pos,
                                     Float cur_val) {
        std::int64_t count = 0;
        if constexpr (!indices_out) {
            for (std::int64_t l = 0; l < k; l++) {
                count += (std::int64_t)(selection.get_data()[row * k + l] == cur_val);
            }
        }
        else {
            for (std::int64_t l = 0; l < k; l++) {
                count += (std::int64_t)(indices.get_data()[row * k + l] == pos);
            }
        }
        REQUIRE(count > 0);
    }
};

//using selection_types = std::tuple<float, double>;
using selection_types = std::tuple<float>;
/* TEMPLATE_LIST_TEST_M(selection_by_rows_test,
                     "selection degenerated test (k == 1)",
                     "[block select][small]",
                     selection_types) {
    using Float = TestType;
    Float data[] = { -2.0f, 5.0f, -3.0f, 3.0f, 4.0f, 1.0f, 1.0f, 4.0f,
                     4.0f,  1.0f, 1.0f,  0.0f, 0.0f, 5.0f, 1.0f };
    auto data_array = ndarray<Float, 2>::empty(this->get_queue(), { 3, 5 });
    auto event = this->get_queue().submit([&](sycl::handler& cgh) {
        cgh.memcpy(data_array.get_mutable_data(), data, sizeof(Float) * 15);
    });
    event.wait();
    this->test_selection(data_array, 3, 5, 1);
}

TEMPLATE_LIST_TEST_M(selection_by_rows_test,
                     "selection single row test (k == 2)",
                     "[block select][small]",
                     selection_types) {
    using Float = TestType;
    Float data[] = { 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 4.0f,
                     0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 5.0f, 1.0f };
    auto data_array = ndarray<Float, 2>::empty(this->get_queue(), { 1, 15 });
    auto event = this->get_queue().submit([&](sycl::handler& cgh) {
        cgh.memcpy(data_array.get_mutable_data(), data, sizeof(Float) * 15);
    });
    event.wait();
    this->test_selection(data_array, 1, 15, 2);
}

TEMPLATE_LIST_TEST_M(selection_by_rows_test,
                     "selection degenerated test (k == m)",
                     "[block select][small]",
                     selection_types) {
    using Float = TestType;
    Float data[] = { 7.0f, 5.0f, 0.0f, 0.0f,  0.0f, 1.0f, 1.0f, 4.0f,
                     0.0f, 0.0f, 1.0f, -1.0f, 0.0f, 5.0f, 1.0f };
    auto data_array = ndarray<Float, 2>::empty(this->get_queue(), { 3, 5 });
    auto event = this->get_queue().submit([&](sycl::handler& cgh) {
        cgh.memcpy(data_array.get_mutable_data(), data, sizeof(Float) * 15);
    });
    event.wait();

    this->test_selection(data_array, 3, 5, 5);
}

TEMPLATE_LIST_TEST_M(selection_by_rows_test,
                     "selection test (all zeroes)",
                     "[block select][small]",
                     selection_types) {
    using Float = TestType;
    auto [data_array, event] = ndarray<Float, 2>::zeros(this->get_queue(), { 3, 5 });
    event.wait();
    this->test_selection(data_array, 3, 5, 2);
}

TEMPLATE_LIST_TEST_M(selection_by_rows_test,
                     "selection test on random data",
                     "[block select][small]",
                     selection_types) {
    using Float = TestType;
    std::int64_t rows = 17;
    std::int64_t cols = 33;
    const auto df = GENERATE_DATAFRAME(te::dataframe_builder{ rows, cols }.fill_uniform(-0.2, 0.5));
    const table df_table = df.get_table(this->get_homogen_table_id());
    const auto df_rows = row_accessor<const Float>(df_table).pull(this->get_queue(), { 0, -1 });
    auto data_array = ndarray<Float, 2>::wrap(df_rows.get_data(), { rows, cols });
    this->test_selection(data_array, rows, cols, 31);
}
*/
TEMPLATE_LIST_TEST_M(selection_by_rows_test,
                     "selection test on single random row (k > 32)",
                     "[block select][small]",
                     selection_types) {
    using Float = TestType;
    std::int64_t rows = 1;
    std::int64_t cols = 35;
    std::int64_t k = 33;
    const auto df = GENERATE_DATAFRAME(te::dataframe_builder{ rows, cols }.fill_uniform(-0.2, 0.5));
    const table df_table = df.get_table(this->get_homogen_table_id());
    const auto df_rows = row_accessor<const Float>(df_table).pull(this->get_queue(), { 0, -1 });
    auto data_array = ndarray<Float, 2>::wrap(df_rows.get_data(), { rows, cols });
    this->test_selection(data_array, rows, cols, k);
}

TEMPLATE_LIST_TEST_M(selection_by_rows_test,
                     "selection test on random block (k > 32)",
                     "[block select][small]",
                     selection_types) {
    using Float = TestType;
    std::int64_t rows = 17;
    std::int64_t cols = 35;
    std::int64_t k = 33;
    const auto df = GENERATE_DATAFRAME(te::dataframe_builder{ rows, cols }.fill_uniform(-0.2, 0.5));
    const table df_table = df.get_table(this->get_homogen_table_id());
    const auto df_rows = row_accessor<const Float>(df_table).pull(this->get_queue(), { 0, -1 });
    auto data_array = ndarray<Float, 2>::wrap(df_rows.get_data(), { rows, cols });
    this->test_selection(data_array, rows, cols, k);
}
} // namespace oneapi::dal::backend::primitives::test
