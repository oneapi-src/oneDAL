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

#include <iomanip>
#include <iostream>
#include <chrono>
#include <random>

#include "oneapi/dal/algo/knn.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;

using Float = float;
// using Float = float;

int main(int argc, char const *argv[]) {
    // const auto train_data_file_name = get_data_path("x_ref.csv");
    // const auto query_data_file_name = get_data_path("x_query.csv");

    // const auto x_train = dal::read<dal::table>(dal::csv::data_source{ train_data_file_name });
    // const auto x_query = dal::read<dal::table>(dal::csv::data_source{ query_data_file_name });

    std::default_random_engine state(777);
    std::normal_distribution<Float> normal(-10.0, 10.0);

    const std::size_t rows_train_count = (argc == 1) ? 1000000 : std::stoi(argv[1]);
    const std::size_t columns_count = (argc == 2) ? 512 : std::stoi(argv[2]);
    const std::size_t rows_query_count = 100;

    auto x_train_arr = dal::array<Float>::empty(rows_train_count * columns_count);
    auto x_train_data = x_train_arr.get_mutable_data();
    for (std::size_t i = 0; i < rows_train_count * columns_count; ++i) {
        x_train_data[i] = normal(state);
    }
    auto x_train = dal::homogen_table::wrap(x_train_arr, rows_train_count, columns_count);

    auto x_query_arr = dal::array<Float>::empty(rows_query_count * columns_count);
    auto x_query_data = x_query_arr.get_mutable_data();
    for (std::size_t i = 0; i < rows_query_count * columns_count; ++i) {
        x_query_data[i] = normal(state);
    }
    auto x_query = dal::homogen_table::wrap(x_query_arr, rows_query_count, columns_count);

    const auto cosine_desc = dal::cosine_distance::descriptor<Float>{};
    const std::size_t neighbors_count = 10;

    const auto knn_desc =
        dal::knn::descriptor<Float,
                             dal::knn::method::brute_force,
                             dal::knn::task::search,
                             dal::cosine_distance::descriptor<Float>>(2,
                                                                      neighbors_count,
                                                                      cosine_desc);

    auto t11 = std::chrono::high_resolution_clock::now();
    const auto train_result = dal::train(knn_desc, x_train);
    auto t12 = std::chrono::high_resolution_clock::now();

    std::cout << "Time train: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t12 - t11).count()
              << " ms\n";

    auto t21 = std::chrono::high_resolution_clock::now();
    const auto test_result = dal::infer(knn_desc, x_query, train_result.get_model());
    auto t22 = std::chrono::high_resolution_clock::now();

    std::cout << "Time infer: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t21).count()
              << " ms\n\n";

    // std::cout << "Indices results:\n" << test_result.get_indices() << std::endl;
    std::cout << "Distance results:\n" << test_result.get_distances() << std::endl;
    return 0;
}
