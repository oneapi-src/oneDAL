/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#include <sycl/sycl.hpp>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/pca.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;

void run(sycl::queue& q) {
    const auto train_data_file_name = get_data_path("pca_normalized.csv");
    const std::int64_t nBlocks = 2;
    const auto x_train = dal::read<dal::table>(q, dal::csv::data_source{ train_data_file_name });

    dal::pca::partial_train_result<> partial_result;
    const auto pca_desc =
        dal::pca::descriptor<>().set_component_count(5).set_deterministic(true).set_do_scale(false);
    auto input_table = split_table_by_rows<double>(x_train, nBlocks);

    for (std::int64_t i = 0; i < nBlocks; i++) {
        partial_result = dal::partial_train(pca_desc, partial_result, input_table[i]);
    }
    auto result_train = dal::finalize_train(pca_desc, partial_result);

    std::cout << "Eigenvectors:\n" << result_train.get_eigenvectors() << std::endl;

    std::cout << "Eigenvalues:\n" << result_train.get_eigenvalues() << std::endl;

    const auto result_infer = dal::infer(q, pca_desc, result_train.get_model(), x_train);

    std::cout << "Transformed data:\n" << result_infer.get_transformed_data() << std::endl;
}

int main(int argc, char const* argv[]) {
    for (auto d : list_devices()) {
        std::cout << "Running on " << d.get_platform().get_info<sycl::info::platform::name>()
                  << ", " << d.get_info<sycl::info::device::name>() << "\n"
                  << std::endl;
        auto q = sycl::queue{ d };
        run(q);
    }
    return 0;
}
