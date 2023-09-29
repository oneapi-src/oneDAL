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

#include "oneapi/dal/algo/logistic_regression.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "example_util/utils.hpp"

namespace dal = oneapi::dal;
namespace result_options = dal::logistic_regression::result_options;

void run(sycl::queue& q) {
    const auto x_train_filename = get_data_path("df_binary_classification_train_data.csv");
    const auto y_train_filename = get_data_path("df_binary_classification_train_label.csv");
    const auto x_test_filename = get_data_path("df_binary_classification_test_data.csv");
    const auto y_test_filename = get_data_path("df_binary_classification_test_label.csv");
    const auto params_filename = get_data_path("logreg_params.csv");

    const auto x_train = dal::read<dal::table>(dal::csv::data_source{ x_train_filename });
    const auto y_train = dal::read<dal::table>(dal::csv::data_source{ y_train_filename });
    const auto x_test = dal::read<dal::table>(dal::csv::data_source{ x_test_filename });
    const auto y_test = dal::read<dal::table>(dal::csv::data_source{ y_test_filename });

    const auto log_reg_desc = dal::logistic_regression::descriptor<>(true, 2.0).set_result_options(
        result_options::coefficients | result_options::intercept);

    const auto train_result = dal::train(q, log_reg_desc, x_train, y_train);

    std::cout << "Coefficients:\n" << train_result.get_coefficients() << std::endl;
    std::cout << "Intercept:\n" << train_result.get_intercept() << std::endl;

    const auto log_reg_model = train_result.get_model();

    const auto test_result = dal::infer(q, log_reg_desc, x_test, log_reg_model);

    std::cout << "Test results:\n" << test_result.get_responses() << std::endl;
    std::cout << "True responses:\n" << y_test << std::endl;

    auto y_true_arr = oneapi::dal::row_accessor<const std::int32_t>(y_test).pull();
    const auto gth_ptr = y_true_arr.get_data();

    auto pred_arr =
        oneapi::dal::row_accessor<const std::int32_t>(test_result.get_responses()).pull();
    const auto pred_ptr = pred_arr.get_data();

    std::int64_t acc = 0;

    for (std::int64_t i = 0; i < y_test.get_row_count(); ++i) {
        if (pred_ptr[i] == gth_ptr[i]) {
            acc += 1;
        }
    }

    std::cout << "Accuracy on test: " << double(acc) / y_test.get_row_count() << " (" << acc
              << " out of " << y_test.get_row_count() << ")" << std::endl;
}

int main(int argc, char const* argv[]) {
    for (auto d : list_devices()) {
        std::cout << "Running on " << d.get_platform().get_info<sycl::info::platform::name>()
                  << ", " << d.get_info<sycl::info::device::name>() << "\n"
                  << std::endl;
        auto q = sycl::queue{ d };
        run(q);
    }
    return 0;
}
