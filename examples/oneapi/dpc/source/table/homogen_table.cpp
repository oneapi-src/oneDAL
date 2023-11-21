/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include <sycl/sycl.hpp>
#include <iostream>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include <memory>
#include <iostream>

#include "example_util/utils.hpp"

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/table/homogen.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace dal = oneapi::dal;

template <typename Type = float>
dal::table get_table(sycl::queue& queue, std::int64_t row_count, std::int64_t column_count) {
    const std::int64_t elem_count = row_count * column_count;
    auto* const raw_data = new Type[elem_count];

    // Let's create an array using raw pointer and deleter
    auto data = dal::array<Type>(raw_data,
                                 elem_count, //
                                 [](Type* const ptr) -> void {
                                     delete[] ptr;
                                 });

    for (std::int64_t row = 0l; row < row_count; ++row) {
        for (std::int64_t col = 0l; col < column_count; ++col) {
            const std::int64_t idx = row * column_count + col;
            raw_data[idx] = static_cast<Type>(row * col);
        }
    }

    auto array = to_device(queue, data);

    return dal::homogen_table::wrap(array, row_count, column_count);
}

void run(sycl::queue& queue) {
    constexpr std::int64_t row_count = 4;
    constexpr std::int64_t column_count = 3;

    const dal::table test_table = get_table(queue, row_count, column_count);

    std::cout << "Number of rows in table: " << test_table.get_row_count() << '\n';
    std::cout << "Number of columns in table: " << test_table.get_column_count() << '\n';

    const bool is_homogen = test_table.get_kind() == dal::homogen_table::kind();
    std::cout << "Is homogeneous table: " << is_homogen << '\n';

    dal::row_accessor<const double> accessor{ test_table };

    dal::array<double> slice = accessor.pull(queue, { 1l, 3l });

    std::cout << "Slice of elements: " << slice << std::endl;
}

int main(int argc, char** argv) {
    for (auto d : list_devices()) {
        std::cout << "Running on " << d.get_platform().get_info<sycl::info::platform::name>()
                  << ", " << d.get_info<sycl::info::device::name>() << "\n"
                  << std::endl;
        auto q = sycl::queue{ d };
        run(q);
    }

    return 0;
}
