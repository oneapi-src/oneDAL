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

#include <CL/sycl.hpp>

#include "gtest/gtest.h"
#include "oneapi/dal/algo/decision_forest/train.hpp"
#include "oneapi/dal/algo/decision_forest/infer.hpp"
#include "oneapi/dal/algo/decision_forest/test/utils.hpp"

using namespace oneapi;
namespace df = oneapi::dal::decision_forest;

using df_hist_classifier = df::descriptor<float, df::method::hist, df::task::classification>;
using df_dense_classifier = df::descriptor<float, df::method::dense, df::task::classification>;

TEST(df_bad_arg_tests, test_checks_for_inputs_exceed_int32) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;

    auto selector = sycl::gpu_selector();
    auto queue = sycl::queue(selector);

    auto x_train = sycl::malloc_shared<float>(row_count_train * column_count, queue);
    ASSERT_NE(x_train, nullptr);
    const auto x_train_table =
        dal::homogen_table::wrap(queue, x_train, row_count_train, column_count);

    auto y_train = sycl::malloc_shared<float>(row_count_train, queue);
    ASSERT_NE(y_train, nullptr);
    const auto y_train_table = dal::homogen_table::wrap(queue, y_train, row_count_train, 1);

    ASSERT_THROW((dal::train(queue,
                             df_hist_classifier{}.set_class_count(0xFFFFFFFF),
                             x_train_table,
                             y_train_table)),
                 dal::domain_error);
    ASSERT_THROW((dal::train(queue,
                             df_hist_classifier{}.set_min_observations_in_leaf_node(0xFFFFFFFF),
                             x_train_table,
                             y_train_table)),
                 dal::domain_error);
    ASSERT_THROW((dal::train(queue,
                             df_hist_classifier{}.set_features_per_node(0xFFFFFFFF),
                             x_train_table,
                             y_train_table)),
                 dal::domain_error);
    ASSERT_THROW((dal::train(queue,
                             df_hist_classifier{}.set_max_bins(0xFFFFFFFF),
                             x_train_table,
                             y_train_table)),
                 dal::domain_error);
    ASSERT_THROW((dal::train(queue,
                             df_hist_classifier{}.set_min_bin_size(0xFFFFFFFF),
                             x_train_table,
                             y_train_table)),
                 dal::domain_error);
}

TEST(df_bad_arg_tests, test_overflow_checks_in_train) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;

    auto selector = sycl::gpu_selector();
    auto queue = sycl::queue(selector);

    auto x_train = sycl::malloc_shared<float>(row_count_train * column_count, queue);
    ASSERT_NE(x_train, nullptr);
    const auto x_train_table =
        dal::homogen_table::wrap(queue, x_train, row_count_train, column_count);

    auto y_train = sycl::malloc_shared<float>(row_count_train, queue);
    ASSERT_NE(y_train, nullptr);
    const auto y_train_table = dal::homogen_table::wrap(queue, y_train, row_count_train, 1);

    ASSERT_THROW((dal::train(queue,
                             df_hist_classifier{}.set_tree_count(0x7FFFFFFFFFFFFFFF),
                             x_train_table,
                             y_train_table)),
                 dal::internal_error);
}

TEST(df_bad_arg_tests, set_infer_params_over_int32) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t row_count_test = 3;
    constexpr std::int64_t column_count = 2;

    const float x_train_host[] = { -2.f, -1.f, -1.f, -1.f, -1.f, -2.f,
                                   +1.f, +1.f, +1.f, +2.f, +2.f, +1.f };
    const float y_train_host[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    const float x_test_host[] = { -1.f, -1.f, 2.f, 2.f, 3.f, 2.f };

    auto selector = sycl::gpu_selector();
    auto queue = sycl::queue(selector);

    auto x_train = sycl::malloc_shared<float>(row_count_train * column_count, queue);
    ASSERT_NE(x_train, nullptr);
    std::memcpy(x_train, x_train_host, sizeof(float) * row_count_train * column_count);
    const auto x_train_table =
        dal::homogen_table::wrap(queue, x_train, row_count_train, column_count);

    auto y_train = sycl::malloc_shared<float>(row_count_train, queue);
    ASSERT_NE(y_train, nullptr);
    std::memcpy(y_train, y_train_host, sizeof(float) * row_count_train);
    const auto y_train_table = dal::homogen_table::wrap(queue, y_train, row_count_train, 1);

    auto x_test = sycl::malloc_shared<float>(row_count_test * column_count, queue);
    ASSERT_NE(x_test, nullptr);
    std::memcpy(x_test, x_test_host, sizeof(float) * row_count_test * column_count);
    const auto x_test_table = dal::homogen_table::wrap(queue, x_test, row_count_test, column_count);

    const auto df_train_desc = df_hist_classifier{};

    const auto result_train = dal::train(queue, df_train_desc, x_train_table, y_train_table);

    ASSERT_THROW((dal::infer(queue,
                             df_dense_classifier{}.set_class_count(0xFFFFFFFF),
                             result_train.get_model(),
                             x_test_table)),
                 dal::domain_error);
}

TEST(infer_and_train_cls_kernels_test, can_process_simple_case_default_params) {
    constexpr double accuracy_threshold = 0.05;
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t row_count_test = 3;
    constexpr std::int64_t column_count = 2;

    const float x_train_host[] = { -2.f, -1.f, -1.f, -1.f, -1.f, -2.f,
                                   +1.f, +1.f, +1.f, +2.f, +2.f, +1.f };
    const float y_train_host[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    const float x_test_host[] = { -1.f, -1.f, 2.f, 2.f, 3.f, 2.f };
    const float y_test_host[] = { 0.f, 1.f, 1.f };

    auto selector = sycl::gpu_selector();
    auto queue = sycl::queue(selector);

    auto x_train = sycl::malloc_shared<float>(row_count_train * column_count, queue);
    ASSERT_NE(x_train, nullptr);
    std::memcpy(x_train, x_train_host, sizeof(float) * row_count_train * column_count);
    const auto x_train_table =
        dal::homogen_table::wrap(queue, x_train, row_count_train, column_count);

    auto y_train = sycl::malloc_shared<float>(row_count_train, queue);
    ASSERT_NE(y_train, nullptr);
    std::memcpy(y_train, y_train_host, sizeof(float) * row_count_train);
    const auto y_train_table = dal::homogen_table::wrap(queue, y_train, row_count_train, 1);

    auto x_test = sycl::malloc_shared<float>(row_count_test * column_count, queue);
    ASSERT_NE(x_test, nullptr);
    std::memcpy(x_test, x_test_host, sizeof(float) * row_count_test * column_count);
    const auto x_test_table = dal::homogen_table::wrap(queue, x_test, row_count_test, column_count);

    const auto df_train_desc = df_hist_classifier{};
    const auto df_infer_desc = df_dense_classifier{};

    const auto result_train = dal::train(queue, df_train_desc, x_train_table, y_train_table);
    ASSERT_EQ(!(result_train.get_var_importance().has_data()), true);
    ASSERT_EQ(!(result_train.get_oob_err().has_data()), true);
    ASSERT_EQ(!(result_train.get_oob_err_per_observation().has_data()), true);
    // infer on CPU for now
    const auto result_infer =
        dal::infer(queue, df_infer_desc, result_train.get_model(), x_test_table);

    auto labels_table = result_infer.get_labels();
    ASSERT_EQ(labels_table.has_data(), true);
    ASSERT_EQ(labels_table.get_row_count(), row_count_test);
    ASSERT_EQ(labels_table.get_column_count(), 1);
    ASSERT_EQ(!result_infer.get_probabilities().has_data(), true);

    ASSERT_LE(calculate_classification_error(labels_table, y_test_host), accuracy_threshold);
}

TEST(infer_and_train_cls_kernels_test, can_process_simple_case_non_default_params) {
    constexpr double accuracy_threshold = 0.05;
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t row_count_test = 3;
    constexpr std::int64_t column_count = 2;
    constexpr std::int64_t tree_count = 10;
    constexpr std::int64_t class_count = 2;

    const float x_train_host[] = { -2.f, -1.f, -1.f, -1.f, -1.f, -2.f,
                                   +1.f, +1.f, +1.f, +2.f, +2.f, +1.f };
    const float y_train_host[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    const float x_test_host[] = { -1.f, -1.f, 2.f, 2.f, 3.f, 2.f };
    const float y_test_host[] = { 0.f, 1.f, 1.f };

    auto selector = sycl::gpu_selector();
    auto queue = sycl::queue(selector);

    auto x_train = sycl::malloc_shared<float>(row_count_train * column_count, queue);
    ASSERT_NE(x_train, nullptr);
    std::memcpy(x_train, x_train_host, sizeof(float) * row_count_train * column_count);
    const auto x_train_table =
        dal::homogen_table::wrap(queue, x_train, row_count_train, column_count);

    auto y_train = sycl::malloc_shared<float>(row_count_train, queue);
    ASSERT_NE(y_train, nullptr);
    std::memcpy(y_train, y_train_host, sizeof(float) * row_count_train);
    const auto y_train_table = dal::homogen_table::wrap(queue, y_train, row_count_train, 1);

    auto x_test = sycl::malloc_shared<float>(row_count_test * column_count, queue);
    ASSERT_NE(x_test, nullptr);
    std::memcpy(x_test, x_test_host, sizeof(float) * row_count_test * column_count);
    const auto x_test_table = dal::homogen_table::wrap(queue, x_test, row_count_test, column_count);

    const auto df_train_desc =
        df_hist_classifier{}
            .set_class_count(class_count)
            .set_tree_count(tree_count)
            .set_features_per_node(1)
            .set_min_observations_in_leaf_node(2)
            .set_variable_importance_mode(df::variable_importance_mode::mdi)
            .set_error_metric_mode(df::error_metric_mode::out_of_bag_error |
                                   df::error_metric_mode::out_of_bag_error_per_observation);

    const auto df_infer_desc =
        df_dense_classifier{}
            .set_infer_mode(df::infer_mode::class_labels | df::infer_mode::class_probabilities)
            .set_voting_mode(df::voting_mode::unweighted);

    const auto result_train = dal::train(queue, df_train_desc, x_train_table, y_train_table);
    ASSERT_EQ(result_train.get_model().get_tree_count(), tree_count);
    ASSERT_EQ(result_train.get_model().get_class_count(), class_count);
    ASSERT_EQ(result_train.get_var_importance().has_data(), true);
    ASSERT_EQ(result_train.get_var_importance().get_column_count(), column_count);
    ASSERT_EQ(result_train.get_var_importance().get_row_count(), 1);
    ASSERT_EQ(result_train.get_oob_err().has_data(), true);
    ASSERT_EQ(result_train.get_oob_err().get_row_count(), 1);
    ASSERT_EQ(result_train.get_oob_err().get_column_count(), 1);
    ASSERT_EQ(result_train.get_oob_err_per_observation().has_data(), true);
    ASSERT_EQ(result_train.get_oob_err_per_observation().get_row_count(), row_count_train);
    ASSERT_EQ(result_train.get_oob_err_per_observation().get_column_count(), 1);

    verify_oob_err_vs_oob_err_per_observation(result_train.get_oob_err(),
                                              result_train.get_oob_err_per_observation(),
                                              accuracy_threshold);

    const auto result_infer =
        dal::infer(queue, df_infer_desc, result_train.get_model(), x_test_table);

    auto labels_table = result_infer.get_labels();
    ASSERT_EQ(labels_table.has_data(), true);
    ASSERT_EQ(labels_table.get_row_count(), row_count_test);
    ASSERT_EQ(labels_table.get_column_count(), 1);
    ASSERT_EQ(result_infer.get_probabilities().has_data(), true);
    ASSERT_EQ(result_infer.get_probabilities().get_column_count(), class_count);
    ASSERT_EQ(result_infer.get_probabilities().get_row_count(), row_count_test);

    ASSERT_LE(calculate_classification_error(labels_table, y_test_host), accuracy_threshold);
}

TEST(infer_and_train_cls_kernels_test, can_process_corner_case) {
    constexpr double accuracy_threshold = 0.05;
    constexpr std::int64_t row_count_train = 3;
    constexpr std::int64_t row_count_test = 1;
    constexpr std::int64_t column_count = 1;

    const float x_train_host[] = { -1.f, 2.f, 2.3f };
    const float y_train_host[] = { 0.f, 1.f, 1.f };

    const float x_test_host[] = { 1.f };
    const float y_test_host[] = { 1.f };

    auto selector = sycl::gpu_selector();
    auto queue = sycl::queue(selector);

    auto x_train = sycl::malloc_shared<float>(row_count_train * column_count, queue);
    ASSERT_NE(x_train, nullptr);
    std::memcpy(x_train, x_train_host, sizeof(float) * row_count_train * column_count);
    const auto x_train_table =
        dal::homogen_table::wrap(queue, x_train, row_count_train, column_count);

    auto y_train = sycl::malloc_shared<float>(row_count_train, queue);
    ASSERT_NE(y_train, nullptr);
    std::memcpy(y_train, y_train_host, sizeof(float) * row_count_train);
    const auto y_train_table = dal::homogen_table::wrap(queue, y_train, row_count_train, 1);

    auto x_test = sycl::malloc_shared<float>(row_count_test * column_count, queue);
    ASSERT_NE(x_test, nullptr);
    std::memcpy(x_test, x_test_host, sizeof(float) * row_count_test * column_count);
    const auto x_test_table = dal::homogen_table::wrap(queue, x_test, row_count_test, column_count);

    const auto df_train_desc = df_hist_classifier{}
                                   .set_class_count(2)
                                   .set_tree_count(10)
                                   .set_min_observations_in_leaf_node(8);
    const auto df_infer_desc = df_dense_classifier{}.set_class_count(2);

    const auto result_train = dal::train(queue, df_train_desc, x_train_table, y_train_table);
    // infer on CPU for now
    const auto result_infer =
        dal::infer(queue, df_infer_desc, result_train.get_model(), x_test_table);

    auto labels_table = result_infer.get_labels();
    ASSERT_EQ(labels_table.has_data(), true);
    ASSERT_EQ(labels_table.get_row_count(), row_count_test);
    ASSERT_EQ(labels_table.get_column_count(), 1);

    ASSERT_LE(calculate_classification_error(labels_table, y_test_host), accuracy_threshold);
}
