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

#include "oneapi/dal/algo/kmeans_init.hpp"
#include "example_util/utils.hpp"

using namespace oneapi;

int main(int argc, char const* argv[]) {
    constexpr std::int64_t row_count    = 8;
    constexpr std::int64_t column_count = 7;

    const size_t nClusters = 2;

    const float data[] = { 1.f,  2.f, 3.f,  4.f,  5.f,  6.f,  -5.f,
                           1.f, -1.f, 0.f,  3.f, 1.f, 2.f,  3.f,
                           4.f,  5.f,  6.f,  1.f,  0.f,  0.f, 0.f,
                           1.f,  2.f, 5.f, 2.f,  9.f, 3.f,  2.f,
                           -4.f, 3.f,  0.f,  4.f, 2.f,  7.f,  5.f,
                           4.f, 2.f,  0.f, -4.f, 0.f,  3.f,  -8.f,
                           2.f,  5.f, 5.f,  -6.f, 3.f, 0.f, -9.f,
                           3.f, 1.f,  -3.f, 3.f,  5.f,  1.f,  7.f };

    const auto data_table = dal::homogen_table{ row_count, column_count, data };

    const auto kmeans_init_desc = dal::kmeans_init::descriptor<>()
        .set_cluster_count(nClusters);

    const auto result = dal::train(kmeans_init_desc, data_table);

    std::cout << "Initial cetroids:" << std::endl << result.get_centroids() << std::endl;

    return 0;
}
