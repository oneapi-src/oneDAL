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

#include <iomanip>
#include <iostream>

#include "oneapi/dal/algo/knn.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

using namespace oneapi;

const char train_data_file_name[]  = "k_nearest_neighbors_train_data.csv";
const char train_label_file_name[] = "k_nearest_neighbors_train_label.csv";
const char test_data_file_name[]   = "k_nearest_neighbors_test_data.csv";
const char test_label_file_name[]  = "k_nearest_neighbors_test_label.csv";

int main(int argc, char const *argv[]) {
  const auto x_train_table = dal::read(dal::csv::data_source{get_data_path(train_data_file_name)});
  const auto y_train_table = dal::read(dal::csv::data_source{get_data_path(train_label_file_name)});

  const auto knn_desc =
      dal::knn::descriptor<float, oneapi::dal::knn::method::kd_tree>()
          .set_class_count(5)
          .set_neighbor_count(1)
          .set_data_use_in_model(false);

  const auto train_result = dal::train(knn_desc, x_train_table, y_train_table);

  const auto x_test_table = dal::read(dal::csv::data_source{get_data_path(test_data_file_name)});
  const auto y_true_table = dal::read(dal::csv::data_source{get_data_path(test_label_file_name)});

  const auto test_result =
      dal::infer(knn_desc, x_test_table, train_result.get_model());

  std::cout << "Test results:" << std::endl
            << test_result.get_labels() << std::endl;
  std::cout << "True labels:" << std::endl << y_true_table << std::endl;

  return 0;
}
