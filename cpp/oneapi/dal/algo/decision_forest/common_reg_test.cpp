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
#include "oneapi/dal/algo/decision_forest/infer.hpp"
#include "oneapi/dal/algo/decision_forest/train.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

using namespace oneapi;
namespace df = oneapi::dal::decision_forest;

TEST(df_bad_arg_tests, set_tree_count) {
    ASSERT_THROW(
        (df::descriptor<float, df::task::regression, df::method::hist>{}.set_tree_count(0)),
        dal::domain_error);
    ASSERT_THROW(
        (df::descriptor<float, df::task::regression, df::method::hist>{}.set_tree_count(-1)),
        dal::domain_error);
}

TEST(df_bad_arg_tests, set_min_observations_in_leaf_node) {
    ASSERT_THROW((df::descriptor<float, df::task::regression, df::method::hist>{}
                      .set_min_observations_in_leaf_node(0)),
                 dal::domain_error);
    ASSERT_THROW((df::descriptor<float, df::task::regression, df::method::hist>{}
                      .set_min_observations_in_leaf_node(-1)),
                 dal::domain_error);
}

TEST(df_bad_arg_tests, set_min_observations_in_split_node) {
    ASSERT_THROW((df::descriptor<float, df::task::regression, df::method::hist>{}
                      .set_min_observations_in_split_node(0)),
                 dal::domain_error);
    ASSERT_THROW((df::descriptor<float, df::task::regression, df::method::hist>{}
                      .set_min_observations_in_split_node(-1)),
                 dal::domain_error);
}

TEST(df_bad_arg_tests, set_min_weight_fraction_in_leaf_node) {
    ASSERT_THROW((df::descriptor<float, df::task::regression, df::method::hist>{}
                      .set_min_weight_fraction_in_leaf_node(-0.1)),
                 dal::domain_error);
    ASSERT_THROW((df::descriptor<float, df::task::regression, df::method::hist>{}
                      .set_min_weight_fraction_in_leaf_node(0.6)),
                 dal::domain_error);

    ASSERT_NO_THROW((df::descriptor<float, df::task::regression, df::method::hist>{}
                         .set_min_weight_fraction_in_leaf_node(0.0)));
    ASSERT_NO_THROW((df::descriptor<float, df::task::regression, df::method::hist>{}
                         .set_min_weight_fraction_in_leaf_node(0.5)));
}

TEST(df_bad_arg_tests, set_min_impurity_decrease_in_split_node) {
    ASSERT_THROW((df::descriptor<float, df::task::regression, df::method::hist>{}
                      .set_min_impurity_decrease_in_split_node(-0.1)),
                 dal::domain_error);

    ASSERT_NO_THROW((df::descriptor<float, df::task::regression, df::method::hist>{}
                         .set_min_impurity_decrease_in_split_node(0.0)));
}

TEST(df_bad_arg_tests, set_observations_per_tree_fraction) {
    ASSERT_THROW((df::descriptor<float, df::task::regression, df::method::hist>{}
                      .set_observations_per_tree_fraction(0.0)),
                 dal::domain_error);
    ASSERT_THROW((df::descriptor<float, df::task::regression, df::method::hist>{}
                      .set_observations_per_tree_fraction(1.1)),
                 dal::domain_error);
    ASSERT_THROW((df::descriptor<float, df::task::regression, df::method::hist>{}
                      .set_observations_per_tree_fraction(-0.5)),
                 dal::domain_error);

    ASSERT_NO_THROW((df::descriptor<float, df::task::regression, df::method::hist>{}
                         .set_observations_per_tree_fraction(1.0)));
    ASSERT_NO_THROW((df::descriptor<float, df::task::regression, df::method::hist>{}
                         .set_observations_per_tree_fraction(0.5)));
}

TEST(df_bad_arg_tests, set_features_per_node) {
    ASSERT_THROW(
        (df::descriptor<float, df::task::regression, df::method::hist>{}.set_features_per_node(-1)),
        dal::domain_error);

    ASSERT_NO_THROW(
        (df::descriptor<float, df::task::regression, df::method::hist>{}.set_features_per_node(0)));
    ASSERT_NO_THROW(
        (df::descriptor<float, df::task::regression, df::method::hist>{}.set_features_per_node(2)));
}

TEST(df_bad_arg_tests, set_impurity_threshold) {
    ASSERT_THROW(
        (df::descriptor<float, df::task::regression, df::method::hist>{}.set_impurity_threshold(
            -0.1)),
        dal::domain_error);

    ASSERT_NO_THROW(
        (df::descriptor<float, df::task::regression, df::method::hist>{}.set_impurity_threshold(
            0.0)));
    ASSERT_NO_THROW(
        (df::descriptor<float, df::task::regression, df::method::hist>{}.set_impurity_threshold(
            0.5)));
}

TEST(df_bad_arg_tests, set_max_tree_depth) {
    ASSERT_THROW(
        (df::descriptor<float, df::task::regression, df::method::hist>{}.set_max_tree_depth(-1)),
        dal::domain_error);
    ASSERT_NO_THROW(
        (df::descriptor<float, df::task::regression, df::method::hist>{}.set_max_tree_depth(0)));
    ASSERT_NO_THROW(
        (df::descriptor<float, df::task::regression, df::method::hist>{}.set_max_tree_depth(5)));
}

TEST(df_bad_arg_tests, set_max_leaf_nodes) {
    ASSERT_THROW(
        (df::descriptor<float, df::task::regression, df::method::hist>{}.set_max_leaf_nodes(-1)),
        dal::domain_error);
    ASSERT_NO_THROW(
        (df::descriptor<float, df::task::regression, df::method::hist>{}.set_max_leaf_nodes(0)));
    ASSERT_NO_THROW(
        (df::descriptor<float, df::task::regression, df::method::hist>{}.set_max_leaf_nodes(5)));
}

TEST(df_bad_arg_tests, set_train_data) {
    constexpr std::int64_t row_count_train = 6;

    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    dal::homogen_table x_train_table;
    const auto y_train_table =
        dal::homogen_table{ y_train, row_count_train, 1, dal::empty_delete<const float>() };

    const auto df_desc = df::descriptor<float, df::task::regression, df::method::dense>{};

    ASSERT_THROW(dal::train(df_desc, x_train_table, y_train_table), dal::domain_error);
}

TEST(df_bad_arg_tests, set_train_labels) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f
    };

    const auto x_train_table = dal::homogen_table{ x_train,
                                                   row_count_train,
                                                   column_count,
                                                   dal::empty_delete<const float>() };
    dal::homogen_table y_train_table;

    const auto df_desc = df::descriptor<float, df::task::regression, df::method::dense>{};

    ASSERT_THROW(dal::train(df_desc, x_train_table, y_train_table), dal::domain_error);
}

TEST(df_bad_arg_tests, set_bootstrap) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f
    };
    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    const auto x_train_table = dal::homogen_table{ x_train,
                                                   row_count_train,
                                                   column_count,
                                                   dal::empty_delete<const float>() };
    const auto y_train_table =
        dal::homogen_table{ y_train, row_count_train, 1, dal::empty_delete<const float>() };

    // check bootstrap(false) and mda_raw
    {
        auto df_desc = df::descriptor<float, df::task::regression, df::method::dense>{}
                           .set_bootstrap(false)
                           .set_variable_importance_mode(df::variable_importance_mode::mda_raw);

        ASSERT_THROW(dal::train(df_desc, x_train_table, y_train_table), dal::invalid_argument);
        ASSERT_NO_THROW(dal::train(df_desc.set_bootstrap(true), x_train_table, y_train_table));
    }
    // check bootstrap(false) and mda_raw_scaled
    {
        auto df_desc = df::descriptor<float, df::task::regression, df::method::dense>{}
                           .set_bootstrap(false)
                           .set_variable_importance_mode(df::variable_importance_mode::mda_scaled);

        ASSERT_THROW(dal::train(df_desc, x_train_table, y_train_table), dal::invalid_argument);
        ASSERT_NO_THROW(dal::train(df_desc.set_bootstrap(true), x_train_table, y_train_table));
    }
    // check bootstrap(false) and out_of_bag_error
    {
        auto df_desc = df::descriptor<float, df::task::regression, df::method::dense>{}
                           .set_bootstrap(false)
                           .set_error_metric_mode(df::error_metric_mode::out_of_bag_error);

        ASSERT_THROW(dal::train(df_desc, x_train_table, y_train_table), dal::invalid_argument);
        ASSERT_NO_THROW(dal::train(df_desc.set_bootstrap(true), x_train_table, y_train_table));
    }
    // check bootstrap(false) and out_of_bag_error_per_observation
    {
        auto df_desc =
            df::descriptor<float, df::task::regression, df::method::dense>{}
                .set_bootstrap(false)
                .set_error_metric_mode(df::error_metric_mode::out_of_bag_error_per_observation);

        ASSERT_THROW(dal::train(df_desc, x_train_table, y_train_table), dal::invalid_argument);
        ASSERT_NO_THROW(dal::train(df_desc.set_bootstrap(true), x_train_table, y_train_table));
    }
}

TEST(df_bad_arg_tests, data_rows_matches_labels_rows) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t row_count_train_invalid = 5;
    constexpr std::int64_t column_count = 2;

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f
    };
    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    const auto x_train_table = dal::homogen_table{ x_train,
                                                   row_count_train,
                                                   column_count,
                                                   dal::empty_delete<const float>() };
    const auto y_train_table =
        dal::homogen_table{ y_train, row_count_train_invalid, 1, dal::empty_delete<const float>() };

    const auto df_desc = df::descriptor<float, df::task::regression, df::method::dense>{};
    ASSERT_THROW(dal::train(df_desc, x_train_table, y_train_table), dal::invalid_argument);
}

TEST(df_bad_arg_tests, set_infer_data) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f
    };
    const float y_train[] = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };

    const auto x_train_table = dal::homogen_table{ x_train,
                                                   row_count_train,
                                                   column_count,
                                                   dal::empty_delete<const float>() };
    const auto y_train_table =
        dal::homogen_table{ y_train, row_count_train, 1, dal::empty_delete<const float>() };
    dal::homogen_table x_test_table;

    const auto df_desc = df::descriptor<float, df::task::regression, df::method::dense>{};
    const auto result_train = dal::train(df_desc, x_train_table, y_train_table);

    ASSERT_THROW(dal::infer(df_desc, result_train.get_model(), x_test_table), dal::domain_error);
}
