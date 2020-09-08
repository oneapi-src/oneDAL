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
#include "oneapi/dal/algo/svm/infer.hpp"
#include "oneapi/dal/algo/svm/train.hpp"
#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

using namespace oneapi::dal;
using std::int32_t;

TEST(svm_thunder_dense_test, can_classify_linear_separable_surface) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;
    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f,
    };
    const float y_train[] = {
        -1.f, -1.f, -1.f, +1.f, +1.f, +1.f,
    };
    constexpr std::int64_t support_index_negative = 1;
    constexpr std::int64_t support_index_positive = 3;
    const auto x_train_table = homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = homogen_table::wrap(y_train, row_count_train, 1);

    const auto svm_desc = svm::descriptor{}.set_c(1.0);
    const auto result_train = train(svm_desc, x_train_table, y_train_table);
    ASSERT_EQ(result_train.get_support_vector_count(), 2);

    auto support_indices_table = result_train.get_support_indices();
    const auto support_indices = row_accessor<const float>(support_indices_table).pull();
    ASSERT_EQ(support_indices[0], support_index_negative);
    ASSERT_EQ(support_indices[1], support_index_positive);

    const auto result_infer = infer(svm_desc, result_train.get_model(), x_train_table);
    auto decision_function_table = result_infer.get_decision_function();
    const auto decision_function = row_accessor<const float>(decision_function_table).pull();

    ASSERT_FLOAT_EQ(decision_function[support_index_negative], -1.f);
    ASSERT_FLOAT_EQ(decision_function[support_index_positive], +1.f);
}

TEST(svm_thunder_dense_test, can_classify_linear_separable_surface_with_not_default_linear_kernel) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;
    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f,
    };
    const float y_train[] = {
        -1.f, -1.f, -1.f, +1.f, +1.f, +1.f,
    };
    constexpr std::int64_t support_index_negative = 1;
    constexpr std::int64_t support_index_positive = 3;
    const auto x_train_table = homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = homogen_table::wrap(y_train, row_count_train, 1);

    const auto kernel_desc = linear_kernel::descriptor{}.set_scale(0.1).set_shift(0.0);
    const auto svm_desc = svm::descriptor{ kernel_desc }.set_c(10.0);
    const auto result_train = train(svm_desc, x_train_table, y_train_table);
    ASSERT_EQ(result_train.get_support_vector_count(), 2);

    auto support_indices_table = result_train.get_support_indices();
    const auto support_indices = row_accessor<const float>(support_indices_table).pull();
    ASSERT_EQ(support_indices[0], support_index_negative);
    ASSERT_EQ(support_indices[1], support_index_positive);

    const auto result_infer = infer(svm_desc, result_train.get_model(), x_train_table);
    auto decision_function_table = result_infer.get_decision_function();
    const auto decision_function = row_accessor<const float>(decision_function_table).pull();
    ASSERT_FLOAT_EQ(decision_function[support_index_negative], -1.f);
    ASSERT_FLOAT_EQ(decision_function[support_index_positive], +1.f);
}

TEST(svm_thunder_dense_test, can_classify_linear_separable_surface_with_big_margin) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;
    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f,
    };
    const float y_train[] = {
        -1.f, -1.f, -1.f, +1.f, +1.f, +1.f,
    };
    const auto x_train_table = homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = homogen_table::wrap(y_train, row_count_train, 1);
    const auto svm_desc = svm::descriptor{}.set_c(1e-1);

    const auto result_train = train(svm_desc, x_train_table, y_train_table);
    ASSERT_EQ(result_train.get_support_vector_count(), row_count_train);

    auto support_indices_table = result_train.get_support_indices();
    const auto support_indices = row_accessor<const float>(support_indices_table).pull();
    for (std::int64_t i = 0; i < support_indices.get_count(); i++)
        ASSERT_EQ(support_indices[i], i);

    const auto result_infer = infer(svm_desc, result_train.get_model(), x_train_table);

    auto labels_table = result_infer.get_labels();
    const auto labels = row_accessor<const float>(labels_table).pull();
    for (std::int64_t i = 0; i < row_count_train / 2; i++) {
        ASSERT_FLOAT_EQ(labels[i], -1.f);
    }
    for (std::int64_t i = row_count_train / 2; i < row_count_train; i++) {
        ASSERT_FLOAT_EQ(labels[i], +1.f);
    }
}

TEST(svm_thunder_dense_test, can_classify_linear_not_separable_surface) {
    constexpr std::int64_t row_count_train = 8;
    constexpr std::int64_t column_count = 2;
    const float x_train[] = { -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f,
                              +1.f, +2.f, +2.f, +1.f, -3.f, -3.f, +3.f, +3.f };
    const float y_train[] = { -1.f, -1.f, -1.f, +1.f, +1.f, +1.f, +1.f, -1.f };
    const auto x_train_table = homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = homogen_table::wrap(y_train, row_count_train, 1);

    const auto svm_desc = svm::descriptor{}.set_c(1.0);
    const auto result_train = train(svm_desc, x_train_table, y_train_table);

    const auto result_infer = infer(svm_desc, result_train.get_model(), x_train_table);
    auto labels_table = result_infer.get_labels();
    const auto labels = row_accessor<const float>(labels_table).pull();
    for (int i = 0; i < 3; i++) {
        ASSERT_FLOAT_EQ(labels[i], -1.f);
    }
    for (int i = 3; i < 6; i++) {
        ASSERT_FLOAT_EQ(labels[i], +1.f);
    }
}

TEST(svm_thunder_dense_test, can_classify_quadric_separable_surface_with_rbf_kernel) {
    constexpr std::int64_t row_count_train = 12;
    constexpr std::int64_t column_count = 2;
    const float x_train[] = {
        -2.f, 0.f, -2.f, -1.f,  -2.f, +1.f,  +2.f, 0.f,  +2.f, -1.f,  +2.f, +1.f,
        -1.f, 0.f, -1.f, -0.5f, -1.f, +0.5f, +1.f, 0.5f, +1.f, -0.5f, +1.f, +0.5f,
    };
    const float y_train[] = {
        -1.f, -1.f, -1.f, -1.f, -1.f, -1.f, +1.f, +1.f, +1.f, +1.f, +1.f, +1.f,
    };
    const auto x_train_table = homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = homogen_table::wrap(y_train, row_count_train, 1);

    const auto kernel_desc = rbf_kernel::descriptor{}.set_sigma(1.0);
    const auto svm_desc = svm::descriptor{ kernel_desc }.set_c(1.0);
    const auto result_train = train(svm_desc, x_train_table, y_train_table);
    ASSERT_EQ(result_train.get_support_vector_count(), row_count_train);

    const auto result_infer = infer(svm_desc, result_train.get_model(), x_train_table);

    auto labels_table = result_infer.get_labels();
    const auto labels = row_accessor<const float>(labels_table).pull();
    for (std::int64_t i = 0; i < row_count_train / 2; i++) {
        ASSERT_FLOAT_EQ(labels[i], -1.f);
    }
    for (std::int64_t i = row_count_train / 2; i < row_count_train; i++) {
        ASSERT_FLOAT_EQ(labels[i], +1.f);
    }
}

TEST(svm_thunder_dense_test, can_classify_linear_separable_surface_with_equal_weights) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;
    const float x_train[] = {
        -2.f, 0.f, -1.f, -1.f, 0.f, -2.f, 0.f, +2.f, +1.f, +1.f, +2.f, +0.f,
    };
    const float y_train[] = {
        -1.f, -1.f, -1.f, +1.f, +1.f, +1.f,
    };
    const float weights[] = {
        +1.f, +1.f, +1.f, +1.f, +1.f, +1.f,
    };
    const auto x_train_table = homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = homogen_table::wrap(y_train, row_count_train, 1);
    const auto weights_table = homogen_table::wrap(weights, row_count_train, 1);

    const auto svm_desc = svm::descriptor{}.set_c(0.1);
    const auto result_train = train(svm_desc, x_train_table, y_train_table, weights_table);

    const float x_test[] = {
        -1.f,
        1.f,
    };
    const auto x_test_table = homogen_table::wrap(x_test, 1, column_count);
    const auto result_infer = infer(svm_desc, result_train.get_model(), x_test_table);
    auto decision_function_table = result_infer.get_decision_function();
    const auto decision_function = row_accessor<const float>(decision_function_table).pull();
    ASSERT_NEAR(decision_function[0], 0.f, 1e-6);
}

TEST(svm_thunder_dense_test, can_classify_linear_separable_surface_with_boundary_weights) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;
    const float x_train[] = {
        -2.f, 0.f, -1.f, -1.f, 0.f, -2.f, 0.f, +2.f, +1.f, +1.f, +2.f, +0.f,
    };
    const float y_train[] = {
        -1.f, -1.f, -1.f, +1.f, +1.f, +1.f,
    };
    const float weights[] = {
        +10.f, +0.1f, +0.1f, +0.1f, +.1f, 10.f,
    };
    const auto x_train_table = homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = homogen_table::wrap(y_train, row_count_train, 1);
    const auto weights_table = homogen_table::wrap(weights, row_count_train, 1);

    const auto svm_desc = svm::descriptor{}.set_c(0.1);
    const auto result_train = train(svm_desc, x_train_table, y_train_table, weights_table);

    const float x_test[] = {
        -1.f,
        1.f,
    };
    const auto x_test_table = homogen_table::wrap(x_test, 1, column_count);
    const auto result_infer = infer(svm_desc, result_train.get_model(), x_test_table);
    auto decision_function_table = result_infer.get_decision_function();
    const auto decision_function = row_accessor<const float>(decision_function_table).pull();
    ASSERT_LT(decision_function[0], 0.f);
}

TEST(svm_thunder_dense_test, can_classify_linear_separable_surface_with_center_weights) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;
    const float x_train[] = {
        -2.f, 0.f, -1.f, -1.f, 0.f, -2.f, 0.f, +2.f, +1.f, +1.f, +2.f, +0.f,
    };
    const float y_train[] = {
        -1.f, -1.f, -1.f, +1.f, +1.f, +1.f,
    };
    const float weights[] = {
        +0.1f, +0.1f, +10.f, +10.f, +0.1f, +0.1f,
    };
    const auto x_train_table = homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = homogen_table::wrap(y_train, row_count_train, 1);
    const auto weights_table = homogen_table::wrap(weights, row_count_train, 1);

    const auto svm_desc = svm::descriptor{}.set_c(0.1);
    const auto result_train = train(svm_desc, x_train_table, y_train_table, weights_table);

    const float x_test[] = {
        -1.f,
        1.f,
    };
    const auto x_test_table = homogen_table::wrap(x_test, 1, column_count);
    const auto result_infer = infer(svm_desc, result_train.get_model(), x_test_table);
    auto decision_function_table = result_infer.get_decision_function();
    const auto decision_function = row_accessor<const float>(decision_function_table).pull();
    ASSERT_GT(decision_function[0], 0.f);
}

TEST(svm_thunder_dense_test, can_classify_linear_separable_surface_with_different_type_desc) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;
    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f,
    };
    const float y_train[] = {
        -1.f, -1.f, -1.f, +1.f, +1.f, +1.f,
    };
    const auto x_train_table = homogen_table::wrap(x_train, row_count_train, column_count);
    const auto y_train_table = homogen_table::wrap(y_train, row_count_train, 1);
    const auto svm_desc_train = svm::descriptor<float>{}.set_c(1e-1);

    const auto result_train = train(svm_desc_train, x_train_table, y_train_table);
    ASSERT_EQ(result_train.get_support_vector_count(), row_count_train);

    auto support_indices_table = result_train.get_support_indices();
    const auto support_indices = row_accessor<const float>(support_indices_table).pull();
    for (std::int64_t i = 0; i < support_indices.get_count(); i++)
        ASSERT_EQ(support_indices[i], i);

    const auto svm_desc_infer = svm::descriptor<double>{}.set_c(1e-1);
    const auto result_infer = infer(svm_desc_infer, result_train.get_model(), x_train_table);

    auto labels_table = result_infer.get_labels();
    const auto labels = row_accessor<const float>(labels_table).pull();
    for (std::int64_t i = 0; i < row_count_train / 2; i++) {
        ASSERT_FLOAT_EQ(labels[i], -1.f);
    }
    for (std::int64_t i = row_count_train / 2; i < row_count_train; i++) {
        ASSERT_FLOAT_EQ(labels[i], +1.f);
    }
}

TEST(svm_thunder_dense_test, can_classify_any_two_labels) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t range_count = 4;
    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f,
    };
    const float expected_labels_range[range_count][2] = {
        { -1.f, +1.f },
        { +0.f, +1.f },
        { +0.f, +2.f },
        { -1.f, +0.f },
    };

    const float y_train_range[range_count][row_count_train] = {
        {
            -1.f,
            -1.f,
            -1.f,
            +1.f,
            +1.f,
            +1.f,
        },
        {
            0.f,
            0.f,
            0.f,
            +1.f,
            +1.f,
            +1.f,
        },
        {
            0.f,
            0.f,
            0.f,
            +2.f,
            +2.f,
            +2.f,
        },
        {
            -1.f,
            -1.f,
            -1.f,
            +0.f,
            +0.f,
            +0.f,
        },
    };

    for (std::int64_t i = 0; i < range_count; ++i) {
        const auto y_train = y_train_range[i];
        const auto expected_labels = expected_labels_range[i];
        const auto x_train_table = homogen_table::wrap(x_train, row_count_train, column_count);
        const auto y_train_table = homogen_table::wrap(y_train, row_count_train, 1);
        const auto svm_desc_train = svm::descriptor<float>{}.set_c(1e-1);

        const auto result_train = train(svm_desc_train, x_train_table, y_train_table);
        ASSERT_EQ(result_train.get_support_vector_count(), row_count_train);

        auto support_indices_table = result_train.get_support_indices();
        const auto support_indices = row_accessor<const float>(support_indices_table).pull();
        for (std::int64_t i = 0; i < support_indices.get_count(); i++) {
            ASSERT_EQ(support_indices[i], i);
        }

        ASSERT_EQ(result_train.get_model().get_first_class_label(), expected_labels[0]);
        ASSERT_EQ(result_train.get_model().get_second_class_label(), expected_labels[1]);
    }
}
