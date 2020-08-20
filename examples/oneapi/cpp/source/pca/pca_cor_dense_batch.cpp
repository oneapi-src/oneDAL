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

const char data_file_name[] = "pca_normalized.csv";

int main(int argc, char const *argv[]) {
    const auto data_table = dal::read(dal::csv::data_source{get_data_path(data_file_name)});

    const auto pca_desc = dal::pca::descriptor<>()
        .set_component_count(data_table.get_column_count())
        .set_is_deterministic(true);

    const auto result = dal::train(pca_desc, data_table);

    std::cout << "Eigenvectors:" << std::endl
              << result.get_eigenvectors() << std::endl;

    std::cout << "Eigenvalues:" << std::endl
              << result.get_eigenvalues() << std::endl;

    return 0;
}
