/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "gtest/gtest.h"
#include "oneapi/dal/algo/knn/infer.hpp"
#include "oneapi/dal/algo/knn/train.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

using namespace oneapi;
namespace nn = oneapi::dal::knn;

using knn_methods = testing::Types<nn::method::kd_tree, nn::method::brute_force>;

template <typename Tuple>
class knn_common_bad_arg_tests : public ::testing::Test {};

TYPED_TEST_SUITE_P(knn_common_bad_arg_tests);

TYPED_TEST_P(knn_common_bad_arg_tests, test_set_class_count) {
    ASSERT_THROW(
        (nn::descriptor<float, TypeParam, nn::task::classification>{ 2, 1 }.set_class_count(-1)),
        dal::domain_error);
    ASSERT_THROW(
        (nn::descriptor<float, TypeParam, nn::task::classification>{ 2, 1 }.set_class_count(1)),
        dal::domain_error);
    ASSERT_NO_THROW(
        (nn::descriptor<float, TypeParam, nn::task::classification>{ 2, 1 }.set_class_count(2)));
}

TYPED_TEST_P(knn_common_bad_arg_tests, test_set_neighbor_count) {
    ASSERT_THROW(
        (nn::descriptor<float, TypeParam, nn::task::classification>{ 2, 1 }.set_neighbor_count(-1)),
        dal::domain_error);
    ASSERT_THROW(
        (nn::descriptor<float, TypeParam, nn::task::classification>{ 2, 1 }.set_neighbor_count(0)),
        dal::domain_error);
    ASSERT_NO_THROW(
        (nn::descriptor<float, TypeParam, nn::task::classification>{ 2, 1 }.set_neighbor_count(1)));
}

TYPED_TEST_P(knn_common_bad_arg_tests, throws_if_class_count_leads_to_overflow) {
    constexpr std::int64_t row_count_train = 6;

    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    dal::homogen_table x_train_table;
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 1);

    ASSERT_THROW(dal::train(nn::descriptor<float, TypeParam, nn::task::classification>{ 2, 1 }
                                .set_class_count(0xFFFFFFFF),
                            x_train_table,
                            y_train_table),
                 dal::domain_error);
}

TYPED_TEST_P(knn_common_bad_arg_tests, throws_if_neighbor_count_leads_to_overflow) {
    constexpr std::int64_t row_count_train = 6;

    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    dal::homogen_table x_train_table;
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 1);

    ASSERT_THROW(
        dal::train(nn::descriptor<float, nn::method::kd_tree, nn::task::classification>{ 2, 1 }
                       .set_neighbor_count(0xFFFFFFFF),
                   x_train_table,
                   y_train_table),
        dal::domain_error);
}

TYPED_TEST_P(knn_common_bad_arg_tests, throws_if_x_train_table_is_empty) {
    constexpr std::int64_t row_count_train = 6;

    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    dal::homogen_table x_train_table;
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 1);

    const auto knn_desc =
        nn::descriptor<float, nn::method::kd_tree, nn::task::classification>{ 2, 1 };

    ASSERT_THROW(dal::train(knn_desc, x_train_table, y_train_table), dal::domain_error);
}

TYPED_TEST_P(knn_common_bad_arg_tests, throws_if_y_train_table_is_empty) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f
    };

    const auto x_train_table = dal::homogen_table::wrap(x_train, row_count_train, column_count);
    dal::homogen_table y_train_table;

    const auto knn_desc =
        nn::descriptor<float, nn::method::kd_tree, nn::task::classification>{ 2, 1 };
    ASSERT_THROW(dal::train(knn_desc, x_train_table, y_train_table), dal::domain_error);
}

TYPED_TEST_P(knn_common_bad_arg_tests, throws_if_y_train_table_contains_multiple_columns) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f
    };
    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    const auto x_train_table = dal::homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 2);

    const auto knn_desc =
        nn::descriptor<float, nn::method::kd_tree, nn::task::classification>{ 2, 1 };
    ASSERT_THROW(dal::train(knn_desc, x_train_table, y_train_table), dal::domain_error);
}

TYPED_TEST_P(knn_common_bad_arg_tests, throws_if_data_rows_dont_match_for_x_and_y_train_tables) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t row_count_train_invalid = 5;
    constexpr std::int64_t column_count = 2;

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f
    };
    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    const auto x_train_table = dal::homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train_invalid, 1);

    const auto knn_desc =
        nn::descriptor<float, nn::method::kd_tree, nn::task::classification>{ 2, 1 };
    ASSERT_THROW(dal::train(knn_desc, x_train_table, y_train_table), dal::domain_error);
}
REGISTER_TYPED_TEST_SUITE_P(knn_common_bad_arg_tests,
                            test_set_class_count,
                            test_set_neighbor_count,
                            throws_if_class_count_leads_to_overflow,
                            throws_if_neighbor_count_leads_to_overflow,
                            throws_if_x_train_table_is_empty,
                            throws_if_y_train_table_is_empty,
                            throws_if_y_train_table_contains_multiple_columns,
                            throws_if_data_rows_dont_match_for_x_and_y_train_tables);
INSTANTIATE_TYPED_TEST_SUITE_P(run_knn_common_bad_arg_tests, knn_common_bad_arg_tests, knn_methods);

TEST(knn_kd_tree_bad_arg_tests, throws_if_x_test_table_is_empty) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f
    };
    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    const auto x_train_table = dal::homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 1);
    dal::homogen_table x_test_table;

    const auto knn_desc =
        nn::descriptor<float, nn::method::kd_tree, nn::task::classification>{ 2, 1 };
    const auto result_train = dal::train(knn_desc, x_train_table, y_train_table);

    ASSERT_THROW(dal::infer(knn_desc, x_test_table, result_train.get_model()), dal::domain_error);
}

TEST(knn_kd_tree_unit_tests, train_no_throw) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f
    };
    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    const auto x_train_table = dal::homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 1);

    const auto knn_desc =
        nn::descriptor<float, nn::method::kd_tree, nn::task::classification>{ 2, 1 };
    ASSERT_NO_THROW(dal::train(knn_desc, x_train_table, y_train_table));
}

TEST(knn_kd_tree_unit_tests, train_and_infer_no_throw) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f
    };
    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    const auto x_train_table = dal::homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 1);

    const auto knn_desc =
        nn::descriptor<float, nn::method::kd_tree, nn::task::classification>{ 2, 1 };
    const auto result_train = dal::train(knn_desc, x_train_table, y_train_table);

    ASSERT_NO_THROW(dal::infer(knn_desc, x_train_table, result_train.get_model()));
}
