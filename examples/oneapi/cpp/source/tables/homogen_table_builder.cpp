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

#include "oneapi/dal/table/accessor.hpp"

#include "example_util/utils.hpp"

using namespace oneapi;

int main(int argc, char const *argv[]) {
    constexpr std::int64_t row_count = 5;
    constexpr std::int64_t column_count = 3;

    dal::homogen_table_builder builder;
    builder.set_data_type(dal::data_type::float32)
           .allocate(row_count, column_count);

    {
        dal::row_accessor<float> acc { builder };
        auto rows = acc.pull();

        rows.need_mutable_data();
        for(std::int64_t i = 0; i < rows.get_count(); i++) {
            rows[i] = float(i);
        }

        acc.push(rows);
    }

    auto table = builder.build();
    std::cout << "Table values:" << std::endl
              << table << std::endl;
    return 0;
}
