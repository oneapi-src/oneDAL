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

    // create arrays of data, column indices and row offsets of the table
    // in sparse CSR storage format on host
    const float data_host[] = { 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 11.0f, 8.0f };
    const std::int64_t column_indices_host[] = { 1, 2, 4, 3, 2, 4, 2 };
    const std::int64_t row_offsets_host[] = { 1, 4, 5, 7, 8 };

    // allocate SYCL shared memory for storing data, column indices and row offsets arrays
    auto data = sycl::malloc_shared<float>(element_count, q);
    auto column_indices = sycl::malloc_shared<std::int64_t>(element_count, q);
    auto row_offsets = sycl::malloc_shared<std::int64_t>(row_count + 1, q);

    // copy data, column indices and row offsets arrays from host to SYCL shared memory
    q.memcpy(data, data_host, sizeof(float) * element_count).wait();
    q.memcpy(column_indices, column_indices_host, sizeof(std::int64_t) * element_count).wait();
    q.memcpy(row_offsets, row_offsets_host, sizeof(std::int64_t) * (row_count + 1)).wait();

    // create sparse table in CSR format from arrays of data, column indices and row offsets
    // that are allocated in SYCL shared memory
    auto table = dal::csr_table{ q,
                                 data,
                                 column_indices,
                                 row_offsets,
                                 row_count,
                                 column_count,
                                 dal::detail::make_default_delete<const float>(q),
                                 dal::detail::make_default_delete<const std::int64_t>(q),
                                 dal::detail::make_default_delete<const std::int64_t>(q) };
    dal::csr_accessor<const float> acc{ table };

    // pull 2 rows, starting from row number 1, from the sparse table;
    // the pulled rows will have one-based indicies by default
    const auto [subtable_data, subtable_column_indices, subtable_row_offsets] =
        acc.pull(q, { 1, 3 });

    // allocate SYCL shared memory for storing data, column indices and row offsets arrays
    std::unique_ptr<float[]> subtable_data_host(new float[subtable_data.get_count()]);
    std::unique_ptr<std::int64_t[]> subtable_column_indices_host(
        new std::int64_t[subtable_column_indices.get_count()]);
    std::unique_ptr<std::int64_t[]> subtable_row_offsets_host(
        new std::int64_t[subtable_row_offsets.get_count()]);

    // copy data, column indices and row offsets arrays from host to SYCL shared memory
    q.memcpy(subtable_data_host.get(),
             subtable_data.get_data(),
             sizeof(float) * subtable_data.get_count())
        .wait();
    q.memcpy(subtable_column_indices_host.get(),
             subtable_column_indices.get_data(),
             sizeof(std::int64_t) * subtable_column_indices.get_count())
        .wait();
    q.memcpy(subtable_row_offsets_host.get(),
             subtable_row_offsets.get_data(),
             sizeof(std::int64_t) * subtable_row_offsets.get_count())
        .wait();

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
    std::cout << "Values in 2 rows as dense float array:" << std::endl;
    for (std::int64_t i = 0; i < subtable_data.get_count(); i++) {
        std::cout << subtable_data_host[i] << ", ";
    }
    std::cout << std::endl << "Column indices in 2 rows from CSR table:" << std::endl;
    for (std::int64_t i = 0; i < subtable_column_indices.get_count(); i++) {
        std::cout << subtable_column_indices_host[i] << ", ";
    }
    std::cout << std::endl << "Row offsets in 2 rows from CSR table:" << std::endl;
    for (std::int64_t i = 0; i < subtable_row_offsets.get_count(); i++) {
        std::cout << subtable_row_offsets_host[i] << ", ";
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
