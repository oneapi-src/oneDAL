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
#include <numeric>
#include <iostream>
#include <algorithm>

#include "example_util/utils.hpp"

#include "oneapi/dal/table/csr.hpp"
#include "oneapi/dal/table/common.hpp"

#include "oneapi/dal/table/csr_accessor.hpp"

// Generate offsets expecting the table
// to have various amounts of data
// in each row
template <typename Index = std::int64_t>
dal::array<Index> generate_offsets(sycl::queue& queue,
                                   std::int64_t row_count,
                                   std::int64_t column_count) {
    const std::int64_t offset_count = row_count + 1l;
    Index* const raw_data = new Index[offset_count];

    for (std::int64_t row = 0l; row < offset_count; ++row) {
        raw_data[row] = static_cast<Index>(row);
    }

    // Offsets are a cumulative sum of element counts
    std::partial_sum(raw_data, raw_data + offset_count, raw_data);

    auto on_host = dal::array<Index>(raw_data,
                                     offset_count, //
                                     [](Index* const ptr) -> void {
                                         delete[] ptr;
                                     });

    return to_device(queue, on_host);
}

// Generate indices knowing, from the `offsets` array,
// how many elements should be in each row
template <typename Index = std::int64_t>
dal::array<Index> generate_indices(sycl::queue& queue,
                                   std::int64_t column_count,
                                   const dal::array<Index>& offsets_on_device) {
    dal::array<Index> offsets = to_host(offsets_on_device);
    const std::int64_t offset_count = offsets.get_count();
    const std::int64_t row_count = offset_count - 1l;
    const std::int64_t element_count = offsets[row_count];

    Index* const raw_data = new Index[element_count];

    for (std::int64_t row = 0l; row < row_count; ++row) {
        const std::int64_t first = offsets[row];
        const std::int64_t last = offsets[row + 1l];
        const std::int64_t count = last - first;

        // Fill the lower right triangular
        // part of the table (which is non-trivial)
        for (std::int64_t col = 0l; col < count; ++col) {
            const std::int64_t column = column_count - col - 1l;
            raw_data[first + col] = static_cast<Index>(column);
        }

        std::sort(raw_data + first, raw_data + last);
    }

    auto on_host = dal::array<Index>(raw_data,
                                     element_count, //
                                     [](Index* const ptr) -> void {
                                         delete[] ptr;
                                     });

    return to_device(queue, on_host);
}

// Generate data for the table in a deterministic way
template <typename Type = float, typename Index = std::int64_t>
dal::array<Type> generate_data(sycl::queue& queue, const dal::array<Index>& offsets) {
    const std::int64_t offset_count = offsets.get_count();
    const std::int64_t element_count = offsets[offset_count - 1];

    Type* const raw_data = new Type[element_count];

    for (std::int64_t i = 0l; i < element_count; ++i) {
        raw_data[i] = static_cast<Type>(element_count - i);
    }

    auto on_host = dal::array<Type>(raw_data,
                                    element_count, //
                                    [](Type* const ptr) -> void {
                                        delete[] ptr;
                                    });

    return to_device(queue, on_host);
}

void run(sycl::queue& queue) {
    constexpr std::int64_t row_count = 4;
    constexpr std::int64_t column_count = 10;

    // Generate data on a device
    auto offsets = generate_offsets(queue, row_count, column_count);
    auto indices = generate_indices(queue, column_count, offsets);
    auto data = generate_data(queue, offsets);

    // Wrap previously generated pieces of data
    dal::csr_table test_table = dal::csr_table::wrap(data, //
                                                     indices,
                                                     offsets,
                                                     column_count,
                                                     dal::sparse_indexing::zero_based);

    // Sanity checks for the table shape
    std::cout << "Number of rows in table: " << test_table.get_row_count() << '\n';
    std::cout << "Number of columns in table: " << test_table.get_column_count() << '\n';

    // Check the type of abstract table
    const bool is_csr = test_table.get_kind() == dal::csr_table::kind();
    std::cout << "Is CSR table: " << is_csr << '\n';

    // Extract CSR slice of data on the device
    dal::csr_accessor<const float> accessor{ test_table };
    auto [slice_data, slice_indices, slice_offsets] = accessor.pull(queue, { 1, 3 });

    // Move data to be readable on CPU
    dal::array<float> data_on_host = to_host(slice_data);
    std::cout << "Slice of elements (compressed): " << data_on_host << '\n';

    // Move indices to be readable on CPU
    dal::array<std::int64_t> indices_on_host = to_host(slice_indices);
    std::cout << "Slice of indices (compressed): " << indices_on_host << '\n';

    // Move offsets to be readable on CPU
    dal::array<std::int64_t> offsets_on_host = to_host(slice_offsets);
    std::cout << "Slice of offsets: " << offsets_on_host << std::endl;
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
