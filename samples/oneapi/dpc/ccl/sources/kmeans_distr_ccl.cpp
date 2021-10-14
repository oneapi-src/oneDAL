/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/spmd/ccl/communicator.hpp"

#include "utils.hpp"

namespace dal = oneapi::dal;

template <typename Float>
std::vector<dal::table> split_table_by_rows(sycl::queue& queue,
                                            const dal::table& t,
                                            std::int64_t split_count) {
    ONEDAL_ASSERT(split_count > 0);
    ONEDAL_ASSERT(split_count <= t.get_row_count());

    const std::int64_t row_count = t.get_row_count();
    const std::int64_t column_count = t.get_column_count();
    const std::int64_t block_size_regular = row_count / split_count;
    const std::int64_t block_size_tail = row_count % split_count;

    std::vector<dal::table> result(split_count);

    std::int64_t row_offset = 0;
    for (std::int64_t i = 0; i < split_count; i++) {
        const std::int64_t tail = std::int64_t(i + 1 == split_count) * block_size_tail;
        const std::int64_t block_size = block_size_regular + tail;

        const auto row_range = dal::range{ row_offset, row_offset + block_size };
        const auto block = dal::row_accessor<const Float>{ t }.pull(queue, row_range, sycl::usm::alloc::device);
        result[i] = dal::homogen_table::wrap(block, block_size, column_count);
        row_offset += block_size;
    }

    return result;
}

void run(sycl::queue& queue) {
    const auto train_data_file_name = get_data_path("data/kmeans_dense_train_data.csv");
    const auto initial_centroids_file_name = get_data_path("data/kmeans_dense_train_centroids.csv");

    const auto x_train = dal::read<dal::table>(queue, dal::csv::data_source{ train_data_file_name });
    const auto initial_centroids =
        dal::read<dal::table>(queue, dal::csv::data_source{ initial_centroids_file_name });

    const auto kmeans_desc = dal::kmeans::descriptor<>()
                                 .set_cluster_count(20)
                                 .set_max_iteration_count(5)
                                 .set_accuracy_threshold(0.001);
    auto comm = dal::preview::spmd::make_communicator<dal::preview::spmd::backend::ccl>(queue);
    auto rank_id = comm.get_rank();
    auto rank_count = comm.get_rank_count();

    auto input_vec = split_table_by_rows<float>(queue, x_train, rank_count);
    dal::kmeans::train_input local_input { input_vec[rank_id], initial_centroids };

    const auto result_train = dal::preview::train(comm, kmeans_desc, local_input);
    if(comm.get_rank() == 0) {
        std::cout << "Iteration count: " << result_train.get_iteration_count() << std::endl;
        std::cout << "Objective function value: " << result_train.get_objective_function_value()
                << std::endl;
        std::cout << "Centroids:\n" << result_train.get_model().get_centroids() << std::endl;
    }
}

int main(int argc, char const *argv[]) {
    ccl::init();
    int status = MPI_Init(nullptr, nullptr);
    if (status != MPI_SUCCESS) {
        throw std::runtime_error{ "Problem occurred during MPI init" };
    }

    auto device = sycl::gpu_selector{}.select_device();
    std::cout << "Running on " << device.get_info<sycl::info::device::name>() << std::endl;
    sycl::queue q{ device };
    run(q);

    status = MPI_Finalize();
    if (status != MPI_SUCCESS) {
        throw std::runtime_error{ "Problem occurred during MPI finalize" };
    }
    return 0;
}
