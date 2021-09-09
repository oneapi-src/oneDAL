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
#include <iomanip>
#include <iostream>
#include <mpi.h>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/detail/mpi/communicator.hpp"
#include "oneapi/dal/detail/spmd_policy.hpp"

#include "utils.hpp"

namespace dal = oneapi::dal;

void run(dal::detail::spmd_policy<dal::detail::data_parallel_policy> &policy) {
    const auto train_data_file_name = get_data_path("data/kmeans_dense_train_data.csv");
    std::cout << train_data_file_name << std::endl;
    const auto initial_centroids_file_name = get_data_path("data/kmeans_dense_train_centroids.csv");

    const auto x_train = dal::read<dal::table>(policy.get_local(), dal::csv::data_source{ train_data_file_name });
    const auto initial_centroids =
        dal::read<dal::table>(policy.get_local(), dal::csv::data_source{ initial_centroids_file_name });

    const auto kmeans_desc = dal::kmeans::descriptor<>()
                                 .set_cluster_count(20)
                                 .set_max_iteration_count(5)
                                 .set_accuracy_threshold(0.001);

    const auto result_train = dal::train(policy, kmeans_desc, x_train, initial_centroids);
	auto comm = policy.get_communicator();
 	if(comm.get_rank() == 0) {
    std::cout << "Iteration count: " << result_train.get_iteration_count() << std::endl;
    std::cout << "Objective function value: " << result_train.get_objective_function_value()
              << std::endl;
    std::cout << "Centroids:\n" << result_train.get_model().get_centroids() << std::endl;
	}
}

int main(int argc, char const *argv[]) {
    int status = MPI_Init(nullptr, nullptr);
    if (status != MPI_SUCCESS) {
        throw std::runtime_error{ "Problem occurred during MPI init" };
    }

    auto device = sycl::gpu_selector{}.select_device();
    std::cout << "Running on " << device.get_info<sycl::info::device::name>() << std::endl;
    auto q = sycl::queue{ device };

    dal::detail::mpi_communicator comm{ MPI_COMM_WORLD };
    dal::detail::data_parallel_policy local_policy{ q };
    dal::detail::spmd_policy spmd_policy{ local_policy, comm };
    
    run(spmd_policy);

    status = MPI_Finalize();
    if (status != MPI_SUCCESS) {
        throw std::runtime_error{ "Problem occurred during MPI finalize" };
    }
    return 0;
}
