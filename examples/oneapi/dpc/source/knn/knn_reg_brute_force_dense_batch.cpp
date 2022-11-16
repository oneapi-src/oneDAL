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

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/knn.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "oneapi/dal/exceptions.hpp"
#include "example_util/utils.hpp"

namespace dal = oneapi::dal;

void run(sycl::queue& q) {
    const auto train_data_file_name = get_data_path("knn_regression_train_data.csv");
    const auto train_response_file_name = get_data_path("knn_regression_train_responses.csv");
    const auto test_data_file_name = get_data_path("knn_regression_test_data.csv");
    const auto test_response_file_name = get_data_path("knn_regression_test_responses.csv");

    const auto x_train = dal::read<dal::table>(q, dal::csv::data_source{ train_data_file_name });
    const auto y_train =
        dal::read<dal::table>(q, dal::csv::data_source{ train_response_file_name });

    using float_t = float;
    using method_t = dal::knn::method::by_default;
    using task_t = dal::knn::task::regression;
    using descriptor_t = dal::knn::descriptor<float_t, method_t, task_t>;

    const auto knn_desc_uniform = descriptor_t(5);
    const auto knn_desc_distance = descriptor_t(5).set_voting_mode(dal::knn::voting_mode::distance);

    const auto x_test = dal::read<dal::table>(q, dal::csv::data_source{ test_data_file_name });
    const auto y_test = dal::read<dal::table>(q, dal::csv::data_source{ test_response_file_name });

    const auto train_result_uniform = dal::train(q, knn_desc_uniform, x_train, y_train);
    const auto train_result_distance = dal::train(q, knn_desc_distance, x_train, y_train);

    const auto test_result_uniform =
        dal::infer(q, knn_desc_uniform, x_test, train_result_uniform.get_model());
    const auto test_result_distance =
        dal::infer(q, knn_desc_distance, x_test, train_result_distance.get_model());

    std::cout << "Test results (uniform regression):\n"
              << test_result_uniform.get_responses() << std::endl;
    std::cout << "Test results (distance regression):\n"
              << test_result_distance.get_responses() << std::endl;
    std::cout << "True responses:\n" << y_test << std::endl;
}

int main(int argc, char const* argv[]) {
    for (auto d : list_devices()) {
        std::cout << "Running on " << d.get_platform().get_info<sycl::info::platform::name>()
                  << ", " << d.get_info<sycl::info::device::name>() << "\n"
                  << std::endl;
        auto q = sycl::queue{ d };
        // TODO: Should be deleted after regression algorithm introduction on CPU
        try {
            run(q);
        }
        catch (const dal::unimplemented& e) {
            std::cout << e.what() << std::endl;
        }
    }
    return 0;
}
