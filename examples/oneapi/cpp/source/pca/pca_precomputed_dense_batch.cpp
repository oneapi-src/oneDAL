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

#include "oneapi/dal/algo/pca.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

namespace dal = oneapi::dal;

template <typename Method>
void run(const dal::table& x_train, const std::string& method_name) {
    const auto pca_desc =
        dal::pca::descriptor<float, Method>().set_component_count(5).set_deterministic(true);

    const auto result_train = dal::train(pca_desc, x_train);

    std::cout << method_name << "\n" << std::endl;

    std::cout << "Eigenvectors:\n" << result_train.get_eigenvectors() << std::endl;

    std::cout << "Eigenvalues:\n" << result_train.get_eigenvalues() << std::endl;

    const auto result_infer = dal::infer(pca_desc, result_train.get_model(), x_train);

    std::cout << "Transformed data:\n" << result_infer.get_transformed_data() << std::endl;
}

int main(int argc, char const* argv[]) {
    const auto cov_data_file_name = get_data_path("precomputed_covariance.csv");
    const auto cor_data_file_name = get_data_path("precomputed_correlation.csv");

    const auto cov_train = dal::read<dal::table>(dal::csv::data_source{ cov_data_file_name });
    const auto cor_train = dal::read<dal::table>(dal::csv::data_source{ cor_data_file_name });

    run<dal::pca::method::precomputed>(cov_train, "PCA precomputed method with covariance matrix");
    run<dal::pca::method::precomputed>(cor_train, "PCA precomputed method with correlation matrix");

    return 0;
}
