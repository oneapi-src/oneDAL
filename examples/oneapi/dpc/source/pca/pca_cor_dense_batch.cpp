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
#include <CL/sycl.hpp>

#define ONEDAL_DATA_PARALLEL
#include "oneapi/dal/algo/pca.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

using namespace oneapi;

void run(sycl::queue& queue) {
    const std::string train_data_file_name = get_data_path("pca_normalized.csv");

    const auto x_train = dal::read<dal::table>(queue, dal::csv::data_source{ train_data_file_name });

    const auto pca_desc = dal::pca::descriptor<>()
        .set_component_count(5)
        .set_deterministic(true);

    const auto result_train = dal::train(queue, pca_desc, x_train);

    std::cout << "Eigenvectors:" << std::endl
              << result_train.get_eigenvectors() << std::endl;

    std::cout << "Eigenvalues:" << std::endl
              << result_train.get_eigenvalues() << std::endl;

    const auto result_infer = dal::infer(queue, pca_desc, result_train.get_model(), x_train);

    std::cout << "Transformed data:" << std::endl
              << result_infer.get_transformed_data() << std::endl;
}

int main(int argc, char const *argv[]) {
    for (auto device : list_devices()) {
        std::cout << "Running on "
                  << device.get_info<sycl::info::device::name>()
                  << std::endl << std::endl;
        auto queue = sycl::queue{device};
        run(queue);
    }
    return 0;
}
