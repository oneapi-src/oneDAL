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
  constexpr std::int64_t row_count_train = 10;
  constexpr std::int64_t row_count_test = 5;
  constexpr std::int64_t column_count = 2;

  const float x_train[] = {
      0.1f,  0.25f, 0.15f, 0.35f, 0.25f, 0.55f, 0.3f, 0.65f, 0.4f, 0.85f,
      0.45f, 0.95f, 0.55f, 1.15f, 0.6f,  1.25f, 0.7f, 1.45f, 0.8f, 1.65f,
  };

  const float y_train[] = {
      0.0079f, 0.0160f, 0.0407f, 0.0573f, 0.0989f,
      0.1240f, 0.1827f, 0.2163f, 0.2919f, 0.3789f,
  };

  const float x_test[] = {
      0.2f, 0.45f, 0.35f, 0.75f, 0.5f, 1.05f, 0.65f, 1.35f, 0.75f, 1.55f,
  };

  const float y_test[] = {
      0.0269f, 0.0767f, 0.1519f, 0.2527f, 0.3340f,
  };
  const auto x_train_table =
      dal::homogen_table{row_count_train, column_count, x_train};
  const auto y_train_table = dal::homogen_table{row_count_train, 1, y_train};

  const auto x_test_table =
      dal::homogen_table{row_count_test, column_count, x_test};
  const auto y_test_table = dal::homogen_table{row_count_test, 1, y_test};

  const auto df_desc =
      df::descriptor<float, df::task::regression, df::method::dense>{}
          .set_tree_count(10)
          .set_features_per_node(1)
          .set_min_observations_in_leaf_node(1)
          .set_variable_importance_mode(df::variable_importance_mode::mdi);

  const auto result_train = dal::train(df_desc, x_train_table, y_train_table,
              df::train_result_to_compute::compute_out_of_bag_error |
              df::train_result_to_compute::
                  compute_out_of_bag_error_per_observation);

  std::cout << "Variable importance results:" << std::endl
            << result_train.get_var_importance() << std::endl;

  std::cout << "OOB error: " << result_train.get_oob_err() << std::endl;
  std::cout << "OOB error per observation:" << std::endl
            << result_train.get_oob_per_observation_err() << std::endl;

  const auto result_infer =
      dal::infer(df_desc, result_train.get_model(), x_test_table, df::infer_result_to_compute::compute_class_labels);

  std::cout << "Prediction results:" << std::endl
            << result_infer.get_labels() << std::endl;

  std::cout << "Ground truth:" << std::endl << y_test_table << std::endl;

  return 0;
}
