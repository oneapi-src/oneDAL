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

#include "example_util/utils.hpp"
#include "oneapi/dal/algo/kmeans_init.hpp"

using namespace oneapi;

int main(int argc, char const *argv[]) {
    constexpr std::int64_t row_count     = 8;
    constexpr std::int64_t column_count  = 2;
    constexpr std::int64_t cluster_count = 2;

    const float x_compute[] = { 1.0,  1.0,  2.0,  2.0,  1.0,  2.0,  2.0,  1.0,
                                -1.0, -1.0, -1.0, -2.0, -2.0, -1.0, -2.0, -2.0 };
    const float x_test[]    = { 1.0, 1.0, 2.0, 2.0 };

    const auto x_train_table = dal::homogen_table::wrap( x_compute, row_count, column_count);
    const auto x_test_table  = dal::homogen_table::wrap( x_test, cluster_count, column_count);

    const auto kmeans_init_desc = dal::kmeans_init::descriptor<>().set_cluster_count(cluster_count);

    const auto result = dal::compute(kmeans_init_desc, x_compute_table);

    std::cout << "Initial cetroids:" << std::endl << result.get_centroids() << std::endl;

    std::cout << "Ground truth:" << std::endl << x_test_table << std::endl;

    return 0;
}
