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

#include "oneapi/dal/algo/decision_forest.hpp"
#include "oneapi/dal/data/accessor.hpp"
#include "oneapi/dal/data/table.hpp"

#include <iomanip>
#include <iostream>

using namespace oneapi;
namespace df = oneapi::dal::decision_forest;

std::ostream &operator<<(std::ostream &stream, const dal::table &table) {
    auto arr     = dal::row_accessor<const float>(table).pull();
    const auto x = arr.get_data();

    for (std::int64_t i = 0; i < table.get_row_count(); i++) {
        for (std::int64_t j = 0; j < table.get_column_count(); j++) {
            std::cout << std::setw(10) << std::setiosflags(std::ios::fixed) << std::setprecision(3)
                      << x[i * table.get_column_count() + j];
        }
        std::cout << std::endl;
    }
    return stream;
}

const std::int64_t tree_count                    = 10;
const std::int64_t features_per_node             = 1;
const std::int64_t min_observations_in_leaf_node = 2;

int main(int argc, char const *argv[]) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t row_count_test  = 3;
    constexpr std::int64_t column_count    = 2;

    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f,
    };
    const float y_train[] = {
        0.f, 0.f, 0.f, 1.f, 1.f, 1.f,
    };

    const float x_test[] = {
        -1.f, -1.f, 2.f, 2.f, 3.f, 2.f,
    };
    const float y_test[] = {
        0.f,
        1.f,
        1.f,
    };

    const auto x_train_table = dal::homogen_table{ row_count_train, column_count, x_train };
    const auto y_train_table = dal::homogen_table{ row_count_train, 1, y_train };

    const auto x_test_table = dal::homogen_table{ row_count_test, column_count, x_test };
    const auto y_test_table = dal::homogen_table{ row_count_test, 1, y_test };

    const auto df_desc =
        df::descriptor<float, df::task::regression, df::method::default_dense>{}
            .set_tree_count(tree_count)
            .set_features_per_node(features_per_node)
            .set_min_observations_in_leaf_node(min_observations_in_leaf_node)
            .set_variable_importance_mode(df::variable_importance_mode::mdi)
            .set_train_results_to_compute(
                (std::uint64_t)df::train_result_to_compute::compute_out_of_bag_error)
            .set_infer_results_to_compute(
                (std::uint64_t)df::infer_result_to_compute::compute_class_labels)
            ;

    const auto result_train = dal::train(df_desc, x_train_table, y_train_table);

    std::cout << "Variable importance results: " << std::endl
              << result_train.get_var_importance() << std::endl;

    std::cout << "OOB error: " << result_train.get_oob_err() << std::endl;

    /* infer part in progress */

    const auto result_infer = dal::infer(df_desc, result_train.get_model(), x_test_table);

    std::cout << "Prediction results: " << std::endl << result_infer.get_prediction() << std::endl;

    std::cout << "Ground truth: " << std::endl << y_test_table << std::endl;

    return 0;
}
