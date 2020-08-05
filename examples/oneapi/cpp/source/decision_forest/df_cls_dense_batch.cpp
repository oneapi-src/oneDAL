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

#include "example_util/utils.hpp"
#include "oneapi/dal/algo/decision_forest.hpp"

using namespace oneapi;
namespace df = oneapi::dal::decision_forest;

int main(int argc, char const *argv[]) {
  constexpr std::int64_t row_count_train = 6;
  constexpr std::int64_t row_count_test = 3;
  constexpr std::int64_t column_count = 2;

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

  const auto x_train_table =
      dal::homogen_table{row_count_train, column_count, x_train};
  const auto y_train_table = dal::homogen_table{row_count_train, 1, y_train};

  const auto x_test_table =
      dal::homogen_table{row_count_test, column_count, x_test};
  const auto y_test_table = dal::homogen_table{row_count_test, 1, y_test};

  const auto df_desc =
      df::descriptor<float, df::task::classification, df::method::dense>{}
          .set_class_count(2)
          .set_tree_count(10)
          .set_features_per_node(1)
          .set_min_observations_in_leaf_node(1)
          .set_variable_importance_mode(df::variable_importance_mode::mdi)
          .set_voting_method(df::voting_method::weighted);

  const auto result_train = dal::train(df_desc, x_train_table, y_train_table, df::train_result_to_compute::compute_out_of_bag_error);

  std::cout << "Variable importance results:" << std::endl
            << result_train.get_var_importance() << std::endl;

  std::cout << "OOB error: " << result_train.get_oob_err() << std::endl;

  const auto result_infer =
      dal::infer(df_desc, result_train.get_model(), x_test_table, df::infer_result_to_compute::compute_class_labels |
              df::infer_result_to_compute::compute_class_probabilities);

  std::cout << "Prediction results:" << std::endl
            << result_infer.get_labels() << std::endl;
  std::cout << "Probabilities results:" << std::endl
            << result_infer.get_probabilities() << std::endl;

  std::cout << "Ground truth:" << std::endl << y_test_table << std::endl;

  return 0;
}
