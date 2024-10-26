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

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include <memory>
#include <iostream>

#include <sycl/sycl.hpp>

#include "example_util/utils.hpp"

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/chunked_array.hpp"
#include "oneapi/dal/table/heterogen.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

namespace dal = oneapi::dal;

// Generate a sequence of numbers
// allocated on a specified device
template <typename Type = float>
dal::array<Type> get_arange(sycl::queue& queue,
                            std::int64_t count,
                            std::int64_t first = 0l,
                            std::int64_t step = 1l) {
    auto* const raw_data = new Type[count];

    for (std::int64_t i = 0l; i < count; ++i) {
        std::int64_t value = step * i + first;
        raw_data[i] = static_cast<Type>(value);
    }

    // Create an array using raw pointer and delete[ ]
    auto on_host = dal::array<Type>(raw_data,
                                    count, //
                                    [](Type* const ptr) -> void {
                                        delete[] ptr;
                                    });

    return to_device(queue, on_host);
}

// Generate a chunked array on a specified queue
// with a specified number of chunks
template <typename Type = float>
dal::chunked_array<Type> get_chunked_arange(sycl::queue& queue,
                                            std::int64_t count,
                                            std::int64_t chunk_count = 2l) {
    dal::chunked_array<Type> chunked_array(chunk_count);

    std::int64_t min_count = count / chunk_count;
    for (std::int64_t i = 0l; i != chunk_count; ++i) {
        std::int64_t first = i * min_count;
        std::int64_t local_count = (i + 1 == chunk_count) ? (count - first) : min_count;
        auto chunk = get_arange<Type>(queue, local_count, first);
        chunked_array.set_chunk(i, chunk);
    }

    return chunked_array;
}

void run(sycl::queue& queue) {
    constexpr std::int64_t row_count = 24;

    // Generate data on the device with different types and
    // numbers of chunks
    auto column_1 = get_chunked_arange<float>(queue, row_count, 1);
    auto column_2 = get_chunked_arange<float>(queue, row_count, 2);
    auto column_3 = get_chunked_arange<std::int8_t>(queue, row_count, 3);
    auto column_4 = get_chunked_arange<std::int16_t>(queue, row_count, 4);
    auto column_5 = get_chunked_arange<std::uint32_t>(queue, row_count, 5);

    // Wrap different columns into a single non-typed
    // heterogeneous table
    dal::table test_table = dal::heterogen_table::wrap( //
        column_1,
        column_2,
        column_3,
        column_4,
        column_5);

    // Sanity checks for the table shape
    std::cout << "Number of rows in table: " << test_table.get_row_count() << '\n';
    std::cout << "Number of columns in table: " << test_table.get_column_count() << '\n';

    // Check the type of abstract table
    const bool is_heterogen = test_table.get_kind() == dal::heterogen_table::kind();
    std::cout << "Is heterogeneous table: " << is_heterogen << '\n';

    // Extract row slice of data on the device
    dal::row_accessor<const float> accessor{ test_table };
    dal::array<float> slice = accessor.pull(queue, { 3l, 17l });

    // Move data to be readable on CPU
    dal::array<float> on_host = to_host(slice);
    std::cout << "Slice of elements: " << on_host << std::endl;
}

int main(int argc, char** argv) {
    // Go through different devices
    for (auto d : list_devices()) {
        std::cout << "Running on " << d.get_platform().get_info<sycl::info::platform::name>()
                  << ", " << d.get_info<sycl::info::device::name>() << "\n"
                  << std::endl;
        auto q = sycl::queue{ d };
        run(q);
    }

    return 0;
}
