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

#include "oneapi/dal/table/csr_accessor.hpp"
#include "oneapi/dal/table/csr.hpp"

#include "example_util/dpc_helpers.hpp"

namespace dal = oneapi::dal;

void run(sycl::queue &q) {
    constexpr std::int64_t row_count = 4;
    constexpr std::int64_t column_count = 4;
    constexpr std::int64_t element_count = 7;

    // create arrays of data, column indices, and row offsets of the table
    // in sparse CSR storage format on host
    const float data_host[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices_host[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t row_offsets_host[] = { 1, 4, 5, 7, 8 };

    // allocate SYCL shared memory for storing data, column indices, and row offset arrays
    auto data = sycl::malloc_shared<float>(element_count, q);
    auto column_indices = sycl::malloc_shared<std::int64_t>(element_count, q);
    auto row_offsets = sycl::malloc_shared<std::int64_t>(row_count + 1, q);

    // copy data, column indices, and row offset arrays from the host to the SYCL shared memory
    auto data_event = q.memcpy(data, data_host, sizeof(float) * element_count);
    auto column_indices_event =
        q.memcpy(column_indices, column_indices_host, sizeof(std::int64_t) * element_count);
    auto row_offsets_event =
        q.memcpy(row_offsets, row_offsets_host, sizeof(std::int64_t) * (row_count + 1));

    // create a sparse table in CSR format from arrays of data, column indices, and row offsets
    // that are allocated in SYCL shared memory
    auto table = dal::csr_table{ q,
                                 data,
                                 column_indices,
                                 row_offsets,
                                 row_count,
                                 column_count,
                                 dal::detail::make_default_delete<const float>(q),
                                 dal::detail::make_default_delete<const std::int64_t>(q),
                                 dal::detail::make_default_delete<const std::int64_t>(q),
                                 dal::sparse_indexing::one_based,
                                 { data_event, column_indices_event, row_offsets_event } };
    dal::csr_accessor<const float> acc{ table };

    // pull the second and third rows of the sparse table
    // the pulled rows have one-based indices by default
    const auto [block_data, block_column_indices, block_row_offsets] = acc.pull(q, { 1, 3 });

    std::cout << "Print the original sparse data table as 3 arrays in CSR storage format:"
              << std::endl;
    std::cout << "Values of the table:" << std::endl;
    for (std::int64_t i = 0; i < element_count; i++) {
        std::cout << data_host[i] << ", ";
    }
    std::cout << std::endl << "Column indices of the table:" << std::endl;
    for (std::int64_t i = 0; i < element_count; i++) {
        std::cout << column_indices_host[i] << ", ";
    }
    std::cout << std::endl << "Row offsets of the table:" << std::endl;
    for (std::int64_t i = 0; i < row_count + 1; i++) {
        std::cout << row_offsets_host[i] << ", ";
    }
    std::cout << std::endl;

    std::cout << std::endl << "Print 2 rows from CSR table as dense float arrays" << std::endl;
    std::cout << "Values in the second and third rows of the table as dense float array:"
              << std::endl;
    for (std::int64_t i = 0; i < block_data.get_count(); i++) {
        std::cout << block_data[i] << ", ";
    }
    std::cout << std::endl
              << "Column indices of the data in the second and third rows from CSR table:"
              << std::endl;
    for (std::int64_t i = 0; i < block_column_indices.get_count(); i++) {
        std::cout << block_column_indices[i] << ", ";
    }
    std::cout << std::endl
              << "Row offsets of the second and third rows from CSR table:" << std::endl;
    for (std::int64_t i = 0; i < block_row_offsets.get_count(); i++) {
        std::cout << block_row_offsets[i] << ", ";
    }
    std::cout << std::endl;
}

int main(int argc, char const *argv[]) {
    for (auto d : list_devices()) {
        std::cout << "Running on " << d.get_platform().get_info<sycl::info::platform::name>()
                  << ", " << d.get_info<sycl::info::device::name>() << "\n"
                  << std::endl;
        auto q = sycl::queue{ d };
        run(q);
    }
    return 0;
}
