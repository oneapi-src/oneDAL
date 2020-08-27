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

#include "oneapi/dal/algo/pca.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

using namespace oneapi;

int main(int argc, char const *argv[]) {
    const std::string data_file_name = get_data_path("pca_normalized.csv");

    const auto data = dal::read<dal::table>(dal::csv::data_source{data_file_name});

    const auto pca_desc = dal::pca::descriptor<>()
        .set_component_count(data.get_column_count())
        .set_is_deterministic(true);

    const auto result = dal::train(pca_desc, data);

    std::cout << "Eigenvectors:" << std::endl
              << result.get_eigenvectors() << std::endl;

    std::cout << "Eigenvalues:" << std::endl
              << result.get_eigenvalues() << std::endl;

    return 0;
}
