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

#include <iostream>

#include "oneapi/dal/table/column_accessor.hpp"
#include "oneapi/dal/table/homogen.hpp"

namespace onedal = oneapi::dal;

int main(int argc, char const *argv[]) {
    constexpr std::int64_t row_count = 6;
    constexpr std::int64_t column_count = 2;
    const float data[] = {
        0.f, 6.f, 1.f, 7.f, 2.f, 8.f, 3.f, 9.f, 4.f, 10.f, 5.f, 11.f,
    };

    auto table = onedal::homogen_table::wrap(data, row_count, column_count);
    onedal::column_accessor<const float> acc{ table };

    for (std::int64_t col = 0; col < table.get_column_count(); col++) {
        std::cout << "column " << col << " values: ";

        const auto col_values = acc.pull(col);
        for (std::int64_t i = 0; i < col_values.get_count(); i++) {
            std::cout << col_values[i] << ", ";
        }
        std::cout << std::endl;
    }

    return 0;
}
