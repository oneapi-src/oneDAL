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
#include "oneapi/dal/backend/primitives/selection/row_partitioning_kernel.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <typename TestType>
class row_partitioning_test : public te::policy_fixture {
public:
    using float_t = TestType;

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<float_t>();
    }

    void test_partitioning(ndview<float_t, 2>& data,
                           std::int64_t start,
                           std::int64_t end,
                           std::int64_t pivot_index) {
        auto row_count = data.get_dimension(0);
        auto col_count = data.get_dimension(1);
        auto data_ptr = data.get_data();

        auto data_tmp = ndarray<float_t, 2>::empty(this->get_queue(), { row_count, col_count });
        auto data_tmp_ptr = data_tmp.get_mutable_data();
        auto cpy_event = this->get_queue().submit([&](sycl::handler& cgh) {
            cgh.memcpy(data_tmp_ptr, data_ptr, sizeof(float_t) * row_count * col_count);
        });
        cpy_event.wait();

        auto split_array = ndarray<int, 2>::empty(this->get_queue(), { row_count, 1 });
        auto index_array = ndarray<int, 2>::empty(this->get_queue(), { row_count, col_count });
        auto split_array_ptr = split_array.get_mutable_data();
        auto index_array_ptr = index_array.get_mutable_data();
        for (int i = 0; i < row_count; i++)
            for (int j = 0; j < col_count; j++)
                index_array_ptr[i * col_count + j] = j;

        auto nd_range2d = get_row_partitioning_range(row_count, col_count);

        auto event = this->get_queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(nd_range2d, [=](sycl::nd_item<2> item) {
                auto sg = item.get_sub_group();
                const int cur_row =
                    item.get_global_id(1) * sg.get_group_range()[0] + sg.get_group_id()[0];
                if (cur_row >= row_count)
                    return;
                int cur_index =
                    row_partitioning_kernel<float_t>(item,
                                                     data_tmp_ptr + col_count * cur_row,
                                                     index_array_ptr + col_count * cur_row,
                                                     start,
                                                     end,
                                                     data_ptr[pivot_index + cur_row * col_count]);
                if (sg.get_local_id()[0] == 0)
                    split_array_ptr[cur_row] = cur_index;
            });
        });
        event.wait();
        check_results(data_tmp, data, index_array, split_array, start, end, pivot_index);
    }

    void check_results(const ndview<float_t, 2>& data,
                       const ndview<float_t, 2>& data_org,
                       const ndview<int, 2>& indices,
                       const ndview<int, 2>& splits,
                       std::int64_t start,
                       std::int64_t end,
                       std::int64_t pivot_index) {
        auto col_count = data.get_dimension(1);
        auto row_count = data.get_dimension(0);

        for (std::int64_t i = 0; i < row_count; i++) {
            std::vector<int> row_indices;
            auto pivot = data_org.get_data()[i * col_count + pivot_index];
            CAPTURE(pivot);
            for (std::int64_t j = 0; j < col_count; j++) {
                auto value = data.get_data()[i * col_count + j];
                auto index = indices.get_data()[i * col_count + j];
                row_indices.push_back(index);
                CAPTURE(i, j);
                if (j < start || j >= end) {
                    CAPTURE(start, end);
                    auto org_value = data_org.get_data()[i * col_count + j];
                    REQUIRE(value == org_value);
                    REQUIRE(indices.get_data()[i * col_count + j] == j);
                    continue;
                }
                auto split = splits.get_data()[i];
                CAPTURE(split);
                if (j < split) {
                    REQUIRE(value < pivot);
                }
                else {
                    REQUIRE(value >= pivot);
                }
                auto index_value = data_org.get_data()[i * col_count + index];
                REQUIRE(value == index_value);
                auto last = std::unique(std::begin(row_indices), std::end(row_indices));
                REQUIRE(last == row_indices.end());
            }
        }
    }
};

using partitioning_types = std::tuple<float, double>;

TEMPLATE_LIST_TEST_M(row_partitioning_test,
                     "row partitioning test on single random row",
                     "[row_partitioning][small]",
                     partitioning_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = TestType;
    constexpr std::int64_t rows = 1;
    constexpr std::int64_t cols = 17;
    std::int64_t pivot_index = 0;

    const auto df = GENERATE_DATAFRAME(te::dataframe_builder{ rows, cols }.fill_uniform(-0.2, 0.5));
    const table df_table = df.get_table(this->get_homogen_table_id());
    const auto df_rows = row_accessor<const float_t>(df_table).pull(this->get_queue(), { 0, -1 });
    auto data_array = ndarray<float_t, 2>::wrap(df_rows.get_data(), { rows, cols });
    this->test_partitioning(data_array, 0, cols, pivot_index);
}

TEMPLATE_LIST_TEST_M(row_partitioning_test,
                     "row partitioning test (two rows)",
                     "[row_partitioning][small]",
                     partitioning_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = TestType;
    constexpr std::int64_t rows = 2;
    constexpr std::int64_t cols = 17;
    std::int64_t pivot_index = 0;

    const auto df = GENERATE_DATAFRAME(te::dataframe_builder{ rows, cols }.fill_uniform(-0.2, 0.5));
    const table df_table = df.get_table(this->get_homogen_table_id());
    const auto df_rows = row_accessor<const float_t>(df_table).pull(this->get_queue(), { 0, -1 });
    auto data_array = ndarray<float_t, 2>::wrap(df_rows.get_data(), { rows, cols });
    this->test_partitioning(data_array, 0, cols, pivot_index);
}

TEMPLATE_LIST_TEST_M(row_partitioning_test,
                     "row partitioning test (unaligned block)",
                     "[row_partitioning][small]",
                     partitioning_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = TestType;
    constexpr std::int64_t rows = 17;
    constexpr std::int64_t cols = 37;
    std::int64_t pivot_index = 0;

    const auto df = GENERATE_DATAFRAME(te::dataframe_builder{ rows, cols }.fill_uniform(-0.2, 0.5));
    const table df_table = df.get_table(this->get_homogen_table_id());
    const auto df_rows = row_accessor<const float_t>(df_table).pull(this->get_queue(), { 0, -1 });
    auto data_array = ndarray<float_t, 2>::wrap(df_rows.get_data(), { rows, cols });
    this->test_partitioning(data_array, 0, cols, pivot_index);
}

TEMPLATE_LIST_TEST_M(row_partitioning_test,
                     "row partitioning test (partial single row)",
                     "[row_partitioning][small]",
                     partitioning_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = TestType;
    constexpr std::int64_t rows = 1;
    constexpr std::int64_t cols = 37;
    std::int64_t start = 1;
    std::int64_t end = 18;
    std::int64_t pivot_index = start;

    const auto df = GENERATE_DATAFRAME(te::dataframe_builder{ rows, cols }.fill_uniform(-0.2, 0.5));
    const table df_table = df.get_table(this->get_homogen_table_id());
    const auto df_rows = row_accessor<const float_t>(df_table).pull(this->get_queue(), { 0, -1 });
    auto data_array = ndarray<float_t, 2>::wrap(df_rows.get_data(), { rows, cols });
    this->test_partitioning(data_array, start, end, pivot_index);
}

TEMPLATE_LIST_TEST_M(row_partitioning_test,
                     "row partitioning test (end of single row)",
                     "[row_partitioning][small]",
                     partitioning_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = TestType;
    constexpr std::int64_t rows = 1;
    constexpr std::int64_t cols = 35;
    std::int64_t start = 26;
    std::int64_t pivot_index = start;

    const auto df = GENERATE_DATAFRAME(te::dataframe_builder{ rows, cols }.fill_uniform(-0.2, 0.5));
    const table df_table = df.get_table(this->get_homogen_table_id());
    const auto df_rows = row_accessor<const float_t>(df_table).pull(this->get_queue(), { 0, -1 });
    auto data_array = ndarray<float_t, 2>::wrap(df_rows.get_data(), { rows, cols });
    this->test_partitioning(data_array, start, cols, pivot_index);
}

TEMPLATE_LIST_TEST_M(row_partitioning_test,
                     "row partitioning test (partial unaligned block)",
                     "[row_partitioning][small]",
                     partitioning_types) {
    SKIP_IF(this->not_float64_friendly());
    using float_t = TestType;
    constexpr std::int64_t rows = 17;
    constexpr std::int64_t cols = 37;
    std::int64_t start = 1;
    std::int64_t end = 19;
    std::int64_t pivot_index = start;

    const auto df = GENERATE_DATAFRAME(te::dataframe_builder{ rows, cols }.fill_uniform(-0.2, 0.5));
    const table df_table = df.get_table(this->get_homogen_table_id());
    const auto df_rows = row_accessor<const float_t>(df_table).pull(this->get_queue(), { 0, -1 });
    auto data_array = ndarray<float_t, 2>::wrap(df_rows.get_data(), { rows, cols });
    this->test_partitioning(data_array, start, end, pivot_index);
}

} // namespace oneapi::dal::backend::primitives::test
