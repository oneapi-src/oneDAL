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
#include "oneapi/dal/data/accessor.hpp"
#include "oneapi/dal/data/table.hpp"

#include "example_util/utils.hpp"

using namespace oneapi;

int main(int argc, char const *argv[]) {
  constexpr std::int64_t row_count = 5;
  constexpr std::int64_t column_count = 3;

  const float x_train[] = {1.f, 2.f, 3.f, 1.f, -1.f, 0.f, 4.f, 5.f,
                           6.f, 1.f, 2.f, 5.f, -4.f, 3.f, 0.f};

  const float y_train[] = {0, 1, 0, 1, 1};

  const auto x_train_table =
      dal::homogen_table{row_count, column_count, x_train};
  const auto y_train_table = dal::homogen_table{row_count, 1, y_train};

  const auto knn_desc =
      dal::knn::descriptor<float, oneapi::dal::knn::method::kd_tree>()
          .set_class_count(2)
          .set_neighbor_count(1)
          .set_data_use_in_model(false);

  const auto train_result = dal::train(knn_desc, x_train_table, y_train_table);

  const float x_test[] = {1.f, 2.f, 2.f, 1.f, -1.f, 1.f, 4.f, 6.f,
                          6.f, 2.f, 2.f, 5.f, -4.f, 3.f, 1.f};

  const float y_true[] = {0, 1, 0, 1, 1};

  const auto x_test_table = dal::homogen_table{row_count, column_count, x_test};
  const auto y_true_table = dal::homogen_table{row_count, 1, y_true};

  const auto test_result =
      dal::infer(knn_desc, x_test_table, train_result.get_model());

  std::cout << "Test results:" << std::endl
            << test_result.get_labels() << std::endl;
  std::cout << "True labels:" << std::endl << y_true_table << std::endl;

  return 0;
}
