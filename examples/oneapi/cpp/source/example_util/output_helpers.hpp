/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#pragma once

#include <iomanip>
#include <iostream>

#include "oneapi/dal/table/row_accessor.hpp"
#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/homogen.hpp"

std::ostream &operator<<(std::ostream &stream, const oneapi::dal::table &table) {
    auto arr = oneapi::dal::row_accessor<const float>(table).pull();
    const auto x = arr.get_data();

    if (table.get_row_count() <= 10) {
        for (std::int64_t i = 0; i < table.get_row_count(); i++) {
            for (std::int64_t j = 0; j < table.get_column_count(); j++) {
                std::cout << std::setw(10) << std::setiosflags(std::ios::fixed)
                          << std::setprecision(3) << x[i * table.get_column_count() + j];
            }
            std::cout << std::endl;
        }
    }
    else {
        for (std::int64_t i = 0; i < 5; i++) {
            for (std::int64_t j = 0; j < table.get_column_count(); j++) {
                std::cout << std::setw(10) << std::setiosflags(std::ios::fixed)
                          << std::setprecision(3) << x[i * table.get_column_count() + j];
            }
            std::cout << std::endl;
        }
        std::cout << "..." << (table.get_row_count() - 10) << " lines skipped..." << std::endl;
        for (std::int64_t i = table.get_row_count() - 5; i < table.get_row_count(); i++) {
            for (std::int64_t j = 0; j < table.get_column_count(); j++) {
                std::cout << std::setw(10) << std::setiosflags(std::ios::fixed)
                          << std::setprecision(3) << x[i * table.get_column_count() + j];
            }
            std::cout << std::endl;
        }
    }
    return stream;
}

void print_table_int(const oneapi::dal::table &table) {
    if (!table.has_data())
        return;
    auto arr = oneapi::dal::row_accessor<const int>(table).pull();
    const auto x = arr.get_data();

    for (std::int64_t i = 0; i < table.get_row_count(); i++) {
        for (std::int64_t j = 0; j < table.get_column_count(); j++) {
            std::cout << x[i * table.get_column_count() + j];
        }
        std::cout << std::endl;
    }
}

void print_table_int_sorted(const oneapi::dal::table &table) {
    if (!table.has_data())
        return;
    auto arr = oneapi::dal::row_accessor<const int>(table).pull();
    const auto x = arr.get_data();

    std::vector<std::vector<int>> v(table.get_row_count());
    for (std::int64_t i = 0; i < table.get_row_count(); i++) {
        for (std::int64_t j = 0; j < table.get_column_count(); j++) {
            v[i].push_back(x[i * table.get_column_count() + j]);
        }
    }

    std::sort(v.begin(), v.end());
    for (auto vi : v) {
        for (std::uint64_t j = 0; j < vi.size() - 1; j++) {
            std::cout << vi[j] << ",";
        }
        std::cout << vi.back();
        std::cout << std::endl;
    }
}
