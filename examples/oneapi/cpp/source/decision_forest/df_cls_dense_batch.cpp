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

std::ostream &operator<<(std::ostream &stream, const dal::table &table) {
  auto arr = dal::row_accessor<const float>(table).pull();
  const auto x = arr.get_data();

  for (std::int64_t i = 0; i < table.get_row_count(); i++) {
    for (std::int64_t j = 0; j < table.get_column_count(); j++) {
      std::cout << std::setw(10) << std::setiosflags(std::ios::fixed)
                << std::setprecision(3) << x[i * table.get_column_count() + j];
    }
    std::cout << std::endl;
  }
  return stream;
}

const std::int64_t n_classes = 2;
const std::int64_t n_trees = 10;
const std::int64_t n_features_per_node = 2;
const std::int64_t min_observations_in_leaf_node = 1;

int main(int argc, char const *argv[]) {

  constexpr std::int64_t row_count_train = 6;
  constexpr std::int64_t column_count = 2;
  const float x_train[] = {
      -2.f, -1.f,
      -1.f, -1.f,
      -1.f, -2.f,
      +1.f, +1.f,
      +1.f, +2.f,
      +2.f, +1.f,
  };
  const float y_train[] = {
      0.f,
      0.f,
      0.f,
      1.f,
      1.f,
      1.f,
  };

  const auto x_train_table =
      dal::homogen_table{row_count_train, column_count, x_train};
  const auto y_train_table =
      dal::homogen_table{row_count_train, 1, y_train};

  const auto decision_forest_train_desc =
      dal::decision_forest::descriptor<float, dal::decision_forest::task::classification,
                           dal::decision_forest::method::default_dense>{}
          .set_n_classes(n_classes)
          .set_n_trees(n_trees)
          .set_features_per_node(n_features_per_node)
          .set_min_observations_in_leaf_node(min_observations_in_leaf_node)
          .set_variable_importance_mode(decision_forest::variable_importance_mode::mdi)
          .set_results_to_compute(decision_forest::result_to_compute_id::compute_out_of_bag_error)

  const auto result_train = dal::train(decision_forest_train_desc, x_train_table, y_train_table);

  std::cout << "Variable importance results: " << std::endl << result_train.get_variable_importance() << std::endl;

  std::cout << "OOB error: " << result_train.get_out_of_bag_error() << std::endl;

  /* infer part in progress */

  return 0;
}
