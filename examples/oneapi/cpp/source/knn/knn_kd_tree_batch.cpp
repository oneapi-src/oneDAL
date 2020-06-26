/*******************************************************************************
* Copyright 2014-2020 Intel Corporation
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

#include "oneapi/dal/data/table.hpp"
#include "oneapi/dal/data/accessor.hpp"
#include "oneapi/dal/algo/knn.hpp"

using namespace oneapi;

std::ostream &operator <<(std::ostream& stream, const dal::table& table) {
    auto arr = dal::row_accessor<const float>(table).pull();
    const auto x = arr.get_data();

    for (std::int64_t i = 0; i < table.get_row_count(); i++) {
        for (std::int64_t j = 0; j < table.get_column_count(); j++) {
            std::cout << std::setw(10)
                      << std::setiosflags(std::ios::fixed)
                      << std::setprecision(3)
                      << x[i * table.get_column_count() + j];
        }
        std::cout << std::endl;
    }
    return stream;
}

int main(int argc, char const *argv[]) {
    constexpr std::int64_t row_count = 5;
    constexpr std::int64_t column_count = 3;

    const float data[] = {
        1.f,  2.f,  3.f,
        1.f,  -1.f, 0.f,
        4.f,  5.f,  6.f,
        1.f,  2.f,  5.f,
        -4.f, 3.f,  0.f
    };

    const float labels[] = {
        0,
        1,
        0,
        1,
        1
    };

    const auto data_table = dal::homogen_table{ row_count, column_count, data };
    const auto label_table = dal::homogen_table{ row_count, 1, labels };

    const auto knn_desc = dal::knn::descriptor<float, oneapi::dal::knn::method::kd_tree>();

    const auto result = dal::train(knn_desc, data_table, label_table);


    return 0;
}
