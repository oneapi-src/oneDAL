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

TEST(knn_bad_arg_tests, set_class_count) {
    ASSERT_THROW((nn::descriptor<float, nn::method::kd_tree, nn::task::classification>{ 2, 1 }
                      .set_class_count(-1)),
                 dal::domain_error);
    ASSERT_THROW((nn::descriptor<float, nn::method::kd_tree, nn::task::classification>{ 2, 1 }
                      .set_class_count(1)),
                 dal::domain_error);
    ASSERT_NO_THROW((nn::descriptor<float, nn::method::kd_tree, nn::task::classification>{ 2, 1 }
                         .set_class_count(2)));
    ASSERT_THROW((nn::descriptor<float, nn::method::brute_force, nn::task::classification>{ 2, 1 }
                      .set_class_count(-1)),
                 dal::domain_error);
    ASSERT_THROW((nn::descriptor<float, nn::method::brute_force, nn::task::classification>{ 2, 1 }
                      .set_class_count(1)),
                 dal::domain_error);
    ASSERT_NO_THROW(
        (nn::descriptor<float, nn::method::brute_force, nn::task::classification>{ 2, 1 }
             .set_class_count(2)));
}

TEST(knn_bad_arg_tests, set_neighbor_count) {
    ASSERT_THROW((nn::descriptor<float, nn::method::kd_tree, nn::task::classification>{ 2, 1 }
                      .set_neighbor_count(-1)),
                 dal::domain_error);
    ASSERT_THROW((nn::descriptor<float, nn::method::kd_tree, nn::task::classification>{ 2, 1 }
                      .set_neighbor_count(0)),
                 dal::domain_error);
    ASSERT_NO_THROW((nn::descriptor<float, nn::method::kd_tree, nn::task::classification>{ 2, 1 }
                         .set_neighbor_count(1)));
    ASSERT_THROW((nn::descriptor<float, nn::method::brute_force, nn::task::classification>{ 2, 1 }
                      .set_neighbor_count(-1)),
                 dal::domain_error);
    ASSERT_THROW((nn::descriptor<float, nn::method::brute_force, nn::task::classification>{ 2, 1 }
                      .set_neighbor_count(0)),
                 dal::domain_error);
    ASSERT_NO_THROW(
        (nn::descriptor<float, nn::method::brute_force, nn::task::classification>{ 2, 1 }
             .set_neighbor_count(1)));
}

TEST(knn_overflow_tests, classes_kd_tree) {
    constexpr std::int64_t row_count_train = 6;

    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    dal::homogen_table x_train_table;
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 1);

    ASSERT_THROW(
        dal::train(nn::descriptor<float, nn::method::kd_tree, nn::task::classification>{ 2, 1 }
                       .set_class_count(0xFFFFFFFF),
                   x_train_table,
                   y_train_table),
        dal::domain_error);
}
/*
TEST(knn_overflow_tests, classes_brute_force) {
    constexpr std::int64_t row_count_train = 6;

    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    dal::homogen_table x_train_table;
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 1);

    ASSERT_THROW(dal::train(nn::descriptor<float,nn::method::brute_force,  nn::task::classification>{2, 1}.set_class_count(0xFFFFFFFF), x_train_table, y_train_table), dal::domain_error);
}
*/
TEST(knn_overflow_tests, neighbors_kd_tree) {
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
/*
TEST(knn_overflow_tests, neighbors_brute_force) {
    constexpr std::int64_t row_count_train = 6;

    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    dal::homogen_table x_train_table;
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 1);

    ASSERT_THROW(dal::train(nn::descriptor<float,nn::method::brute_force,  nn::task::classification>{2, 1}.set_neighbor_count(0xFFFFFFFF), x_train_table, y_train_table), dal::domain_error);
}
*/

TEST(knn_bad_arg_tests, set_train_data_kd_tree) {
    constexpr std::int64_t row_count_train = 6;

    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    dal::homogen_table x_train_table;
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 1);

    const auto knn_desc =
        nn::descriptor<float, nn::method::kd_tree, nn::task::classification>{ 2, 1 };

    ASSERT_THROW(dal::train(knn_desc, x_train_table, y_train_table), dal::domain_error);
}
/*
TEST(knn_bad_arg_tests, set_train_data_brute_force) {
    constexpr std::int64_t row_count_train = 6;

    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    dal::homogen_table x_train_table;
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 1);

    const auto knn_desc =
        nn::descriptor<float,nn::method::brute_force,  nn::task::classification>{2, 1};

    ASSERT_THROW(dal::train(knn_desc, x_train_table, y_train_table), dal::domain_error);
}
*/
TEST(knn_bad_arg_tests, set_train_labels_kd_tree) {
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
/*
TEST(knn_bad_arg_tests, set_train_labels_brute_force) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f
    };

    const auto x_train_table = dal::homogen_table::wrap(x_train, row_count_train, column_count);
    dal::homogen_table y_train_table;

    const auto knn_desc =
        nn::descriptor<float,nn::method::brute_force,  nn::task::classification>{2, 1};
    ASSERT_THROW(dal::train(knn_desc, x_train_table, y_train_table), dal::domain_error);
}
*/
TEST(knn_bad_arg_tests, single_column_labels_kd_tree) {
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
/*
TEST(knn_bad_arg_tests, single_column_labels_brute_force) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f
    };
    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f, 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    const auto x_train_table = dal::homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 2);

    const auto knn_desc =
        nn::descriptor<float,nn::method::brute_force,  nn::task::classification>{2, 1};
    ASSERT_THROW(dal::train(knn_desc, x_train_table, y_train_table), dal::domain_error);
}
*/
TEST(knn_bad_arg_tests, data_rows_matches_labels_rows_kd_tree) {
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
/*
TEST(knn_bad_arg_tests, data_rows_matches_labels_rows_brute_force) {
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
        nn::descriptor<float,nn::method::brute_force,  nn::task::classification>{2, 1};
    ASSERT_THROW(dal::train(knn_desc, x_train_table, y_train_table), dal::domain_error);
}
*/
TEST(knn_bad_arg_tests, bad_infer_data_kd_tree) {
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
/*
TEST(knn_bad_arg_tests, bad_infer_data_brute_force) {
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
        nn::descriptor<float,nn::method::brute_force,  nn::task::classification>{2, 1};
    const auto result_train = dal::train(knn_desc, x_train_table, y_train_table);

    ASSERT_THROW(dal::infer(knn_desc, result_train.get_model(), x_test_table), dal::domain_error);
}
*/
TEST(knn_bad_arg_tests, set_infer_data_kd_tree) {
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
/*
TEST(knn_unit_tests, train_brute_force) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f
    };
    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    const auto x_train_table = dal::homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 1);

    const auto knn_desc =
        nn::descriptor<float,nn::method::brute_force,  nn::task::classification>{2, 1};
    ASSERT_NO_THROW(dal::train(knn_desc, x_train_table, y_train_table), dal::domain_error);
}
*/
TEST(knn_unit_tests, train_kd_tree) {
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

TEST(knn_unit_tests, full_sim_kd_tree) {
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
/*
TEST(knn_unit_tests, full_sim_brute_force) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f
    };
    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    const auto x_train_table = dal::homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 1);

    const auto knn_desc =
        nn::descriptor<float,nn::method::brute_force,  nn::task::classification>{2, 1};
    const auto result_train = dal::train(knn_desc, x_train_table, y_train_table);

    ASSERT_NO_THROW(dal::infer(knn_desc, result_train.get_model(), x_train_table),
                    dal::domain_error);
}
*/