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

#include "gtest/gtest.h"
#include "oneapi/dal/algo/svm/infer.hpp"
#include "oneapi/dal/algo/svm/train.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

using namespace oneapi;
namespace svm = oneapi::dal::svm;

TEST(svm_bad_arg_tests, set_c) {
    ASSERT_THROW((svm::descriptor{}.set_c(0.0)), dal::domain_error);
    ASSERT_THROW((svm::descriptor{}.set_c(-1.0)), dal::domain_error);
}

TEST(svm_bad_arg_tests, set_accuracy_threshold) {
    ASSERT_THROW((svm::descriptor{}.set_accuracy_threshold(-1.0)), dal::domain_error);
}

TEST(svm_bad_arg_tests, set_cache_size) {
    ASSERT_THROW((svm::descriptor{}.set_cache_size(0.0)), dal::domain_error);
    ASSERT_THROW((svm::descriptor{}.set_cache_size(-1.0)), dal::domain_error);
}

TEST(svm_bad_arg_tests, set_tau) {
    ASSERT_THROW((svm::descriptor{}.set_tau(0.0)), dal::domain_error);
    ASSERT_THROW((svm::descriptor{}.set_tau(-1.0)), dal::domain_error);
}

TEST(svm_bad_arg_tests, set_train_data) {
    constexpr std::int64_t row_count_train = 2;

    const float y_train[] = { 0.f, 1.f };
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 1);

    dal::homogen_table x_train_table;

    const auto svm_desc = svm::descriptor{};

    ASSERT_THROW(dal::train(svm_desc, x_train_table, y_train_table), dal::domain_error);
}

TEST(svm_bad_arg_tests, set_train_labels) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f
    };

    const auto x_train_table = dal::homogen_table::wrap(x_train, row_count_train, column_count);
    dal::homogen_table y_train_table;

    const auto svm_desc = svm::descriptor{};

    ASSERT_THROW(dal::train(svm_desc, x_train_table, y_train_table), dal::domain_error);
}

TEST(svm_bad_arg_tests, set_train_weight) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f
    };

    const auto x_train_table = dal::homogen_table::wrap(x_train, row_count_train, column_count);
    dal::homogen_table y_train_table;
    dal::homogen_table weight_train_table;

    const auto svm_desc = svm::descriptor{};

    ASSERT_THROW(dal::train(svm_desc, x_train_table, y_train_table, weight_train_table),
                 dal::domain_error);
}

TEST(svm_bad_arg_tests, data_rows_matches_labels_rows) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t row_count_train_invalid = 5;
    constexpr std::int64_t column_count = 2;

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f
    };
    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    const auto x_train_table = dal::homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train_invalid, 1);

    const auto svm_desc = svm::descriptor{};
    ASSERT_THROW(dal::train(svm_desc, x_train_table, y_train_table), dal::invalid_argument);
}

TEST(svm_bad_arg_tests, set_infer_data) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f
    };
    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    const auto x_train_table = dal::homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = dal::homogen_table::wrap(y_train, row_count_train, 1);
    dal::homogen_table x_test_table;

    const auto svm_desc = svm::descriptor{};
    const auto result_train = dal::train(svm_desc, x_train_table, y_train_table);

    ASSERT_THROW(dal::infer(svm_desc, result_train.get_model(), x_test_table), dal::domain_error);
}
