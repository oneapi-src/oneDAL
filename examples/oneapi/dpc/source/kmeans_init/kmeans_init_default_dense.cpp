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

#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>

#define ONEAPI_DAL_DATA_PARALLEL
#include "oneapi/dal/algo/kmeans_init.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

using namespace oneapi;

const char train_data_file_name[] = "kmeans_init_dense.csv";
const char test_data_file_name[]  = "kmeans_init_dense_ground_truth.csv";

void run(sycl::queue &queue) {
    const auto x_compute_table = dal::read(queue, dal::csv::data_source{get_data_path(train_data_file_name)});
    const auto x_test_table    = dal::read(dal::csv::data_source{get_data_path(test_data_file_name)});;

    const auto kmeans_init_desc = dal::kmeans_init::descriptor<>().set_cluster_count(2);

    const auto result = dal::compute(queue, kmeans_init_desc, x_compute_table);

    std::cout << "Initial cetroids:" << std::endl << result.get_centroids() << std::endl;

    std::cout << "Ground truth:" << std::endl << x_test_table << std::endl;
}

int main(int argc, char const *argv[]) {
    for (auto device : list_devices()) {
        std::cout << "Running on " << device.get_info<sycl::info::device::name>() << std::endl;
        auto queue = sycl::queue{ device };
        run(queue);
    }
    return 0;
}
