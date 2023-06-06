/*******************************************************************************
* Copyright 2022 Intel Corporation
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
    const auto cov_file_name = get_data_path("c_cov.csv");
    const auto data_file_name = get_data_path("c_data.csv");
    const auto means_file_name = get_data_path("c_mns.csv");
    const auto vars_file_name = get_data_path("c_vrs.csv");
    const auto ev_file_name = get_data_path("c_ev.csv");

    const auto x_train = dal::read<dal::table>(q, dal::csv::data_source{ cov_file_name });
    const auto x_test = dal::read<dal::table>(q, dal::csv::data_source{ data_file_name });
    const auto means = dal::read<dal::table>(q, dal::csv::data_source{ means_file_name });
    const auto vars = dal::read<dal::table>(q, dal::csv::data_source{ vars_file_name });
    const auto evs = dal::read<dal::table>(q, dal::csv::data_source{ ev_file_name });
    using float_t = float;
    using method_t = dal::pca::method::precomputed;
    using task_t = dal::pca::task::dim_reduction;
    using descriptor_t = dal::pca::descriptor<float_t, method_t, task_t>;
    const auto pca_desc = descriptor_t().set_component_count(2).set_deterministic(true);

    const auto result_train = dal::train(q, pca_desc, x_train);

    auto model = result_train.get_model();
    model.set_means(means);

    std::cout << "Eigenvectors:\n" << result_train.get_eigenvectors() << std::endl;

    std::cout << "Eigenvalues:\n" << result_train.get_eigenvalues() << std::endl;

    std::cout << "Means:\n" << model.get_means() << std::endl;

    const auto result_infer = dal::infer(q, pca_desc, model, x_test);

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
