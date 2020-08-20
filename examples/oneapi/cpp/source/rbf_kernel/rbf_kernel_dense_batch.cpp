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

#include "oneapi/dal/algo/rbf_kernel.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "example_util/utils.hpp"

using namespace oneapi;

const char data_file_name[] = "kernel_function.csv";

int main(int argc, char const *argv[]) {
    const auto x_table = dal::read(dal::csv::data_source{get_data_path(data_file_name)});
    const auto y_table = dal::read(dal::csv::data_source{get_data_path(data_file_name)});
    const auto kernel_desc = dal::rbf_kernel::descriptor{}.set_sigma(1.0);

    const auto result = dal::compute(kernel_desc, x_table, y_table);

    std::cout << "Values:" << std::endl << result.get_values() << std::endl;

    return 0;
}
