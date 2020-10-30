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

#define ONEDAL_DATA_PARALLEL
#include "oneapi/dal/algo/knn.hpp"
#include "oneapi/dal/io/csv.hpp"

#include "oneapi/dal/exceptions.hpp"
#include "example_util/utils.hpp"

namespace onedal = oneapi::dal;

void run(sycl::queue& q) {
    const auto train_data_file_name = get_data_path("k_nearest_neighbors_train_data.csv");
    const auto train_label_file_name = get_data_path("k_nearest_neighbors_train_label.csv");
    const auto test_data_file_name = get_data_path("k_nearest_neighbors_test_data.csv");
    const auto test_label_file_name = get_data_path("k_nearest_neighbors_test_label.csv");

    const auto x_train =
        onedal::read<onedal::table>(q, onedal::csv::data_source{ train_data_file_name });
    const auto y_train =
        onedal::read<onedal::table>(q, onedal::csv::data_source{ train_label_file_name });

    const auto knn_desc = onedal::knn::descriptor<float,
                                                  onedal::knn::method::brute_force,
                                                  onedal::knn::task::classification>(5, 1);

    const auto x_test =
        onedal::read<onedal::table>(q, onedal::csv::data_source{ test_data_file_name });
    const auto y_test =
        onedal::read<onedal::table>(q, onedal::csv::data_source{ test_label_file_name });

    try {
        const auto train_result = onedal::train(q, knn_desc, x_train, y_train);

        const auto test_result = onedal::infer(q, knn_desc, x_test, train_result.get_model());

        std::cout << "Test results:\n" << test_result.get_labels() << std::endl;
        std::cout << "True labels:\n" << y_test << std::endl;
    }
    catch (onedal::unimplemented& e) {
        std::cout << "  " << e.what() << std::endl;
        return;
    }
}

int main(int argc, char const* argv[]) {
    for (auto d : list_devices()) {
        std::cout << "Running on " << d.get_info<sycl::info::device::name>() << "\n" << std::endl;
        auto q = sycl::queue{ d };
        run(q);
    }
    return 0;
}
