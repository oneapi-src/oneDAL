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
#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/network/mpi.hpp"

using namespace oneapi;

#define MPI

int main(int argc, char* argv[]) {
    int myRank = 0;
    #ifdef MPI
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    #endif

    #ifdef MPI
        std::string train_data_file_name;
        if (myRank == 0)
        {
            train_data_file_name = get_data_path("kmeans_dense_train_data_1.csv");
        }
        else
        {
            train_data_file_name = get_data_path("kmeans_dense_train_data_2.csv");
        }
    #else
        std::string train_data_file_name = get_data_path("kmeans_dense_train_data.csv");
    #endif

    const std::string initial_centroids_file_name = get_data_path("kmeans_dense_train_centroids.csv");
    const std::string test_data_file_name         = get_data_path("kmeans_dense_test_data.csv");
    const std::string test_label_file_name        = get_data_path("kmeans_dense_test_label.csv");

    const auto x_train           = dal::read<dal::table>(dal::csv::data_source{train_data_file_name});
    const auto initial_centroids = dal::read<dal::table>(dal::csv::data_source{initial_centroids_file_name});

    const auto x_test = dal::read<dal::table>(dal::csv::data_source{test_data_file_name});
    const auto y_test = dal::read<dal::table>(dal::csv::data_source{test_label_file_name});

    const auto kmeans_desc = dal::kmeans::descriptor<>()
                                 .set_cluster_count(20)
                                 .set_max_iteration_count(5)
                                 .set_accuracy_threshold(0.001);

    #ifdef MPI
        // TODO: think on better name 
        oneapi::dal::network::mpi::network net;
    #else
        oneapi::dal::network::empty_network net;
    #endif

    const auto result_train = dal::train(kmeans_desc, x_train, initial_centroids, net);

    std::cout << "[" << myRank << "]" << "Iteration count: " << result_train.get_iteration_count() << std::endl;
    std::cout << "[" << myRank << "]" << "Objective function value: " << result_train.get_objective_function_value()
              << std::endl;
    std::cout << "[" << myRank << "]" << "Centroids:" << std::endl << result_train.get_model().get_centroids() << std::endl;

    const auto result_test = dal::infer(kmeans_desc, result_train.get_model(), x_test);

    #ifdef MPI
        MPI_Finalize();
    #endif

    return 0;
}
