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
#include <iomanip>
#include <iostream>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/logistic_regression.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/spmd/ccl/communicator.hpp"

#include "utils.hpp"

namespace dal = oneapi::dal;
namespace lr = dal::logistic_regression;
namespace result_options = dal::logistic_regression::result_options;

void run(sycl::queue &queue) {
    const auto train_data_file_name = get_data_path("df_binary_classification_train_data.csv");
    const auto train_response_file_name = get_data_path("df_binary_classification_train_label.csv");
    const auto test_data_file_name = get_data_path("df_binary_classification_test_data.csv");
    const auto test_response_file_name = get_data_path("df_binary_classification_test_label.csv");

    const auto x_train =
        dal::read<dal::table>(queue, dal::csv::data_source{ train_data_file_name });
    const auto y_train =
        dal::read<dal::table>(queue, dal::csv::data_source{ train_response_file_name });
    const auto x_test = dal::read<dal::table>(queue, dal::csv::data_source{ test_data_file_name });
    const auto y_test =
        dal::read<dal::table>(queue, dal::csv::data_source{ test_response_file_name });

    auto comm = dal::preview::spmd::make_communicator<dal::preview::spmd::backend::ccl>(queue);
    auto rank_id = comm.get_rank();
    auto rank_count = comm.get_rank_count();

    auto x_train_vec = split_table_by_rows<float>(queue, x_train, rank_count);
    auto y_train_vec = split_table_by_rows<float>(queue, y_train, rank_count);
    auto x_test_vec = split_table_by_rows<float>(queue, x_test, rank_count);
    auto y_test_vec = split_table_by_rows<float>(queue, y_test, rank_count);

    using method_t = dal::logistic_regression::method::dense_batch;
    using task_t = dal::logistic_regression::task::classification;
    using optimizer_t = dal::newton_cg::descriptor<>;

    const auto optimizer_desc = dal::newton_cg::descriptor<>(1e-4, 10l);

    const auto log_reg_desc =
        dal::logistic_regression::descriptor<float, method_t, task_t, optimizer_t>(true,
                                                                                   0.5,
                                                                                   optimizer_desc)
            .set_result_options(result_options::coefficients | result_options::intercept |
                                result_options::iterations_count);

    const auto result_train =
        dal::preview::train(comm, log_reg_desc, x_train_vec.at(rank_id), y_train_vec.at(rank_id));

    const auto result_infer =
        dal::preview::infer(comm, log_reg_desc, x_test_vec.at(rank_id), result_train.get_model());

    if (comm.get_rank() == 0) {
        std::cout << "Prediction results:\n" << result_infer.get_responses() << std::endl;

        std::cout << "Ground truth:\n" << y_test_vec.at(rank_id) << std::endl;
    }
}

int main(int argc, char const *argv[]) {
    ccl::init();
    int status = MPI_Init(nullptr, nullptr);
    if (status != MPI_SUCCESS) {
        throw std::runtime_error{ "Problem occurred during MPI init" };
    }

    auto device = sycl::device(sycl::gpu_selector_v);
    std::cout << "Running on " << device.get_platform().get_info<sycl::info::platform::name>()
              << ", " << device.get_info<sycl::info::device::name>() << std::endl;
    sycl::queue q{ device };
    run(q);

    status = MPI_Finalize();
    if (status != MPI_SUCCESS) {
        throw std::runtime_error{ "Problem occurred during MPI finalize" };
    }
    return 0;
}
