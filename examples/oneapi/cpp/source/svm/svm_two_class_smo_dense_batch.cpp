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

#include "oneapi/dal/algo/svm.hpp"

#include "example_util/utils.hpp"

using namespace oneapi;

int main(int argc, char const *argv[]) {
    constexpr std::int64_t row_count_train = 6;
    constexpr std::int64_t column_count = 2;
    const float x_train[] = {
        -2.f, -1.f, -1.f, -1.f, -1.f, -2.f, +1.f, +1.f, +1.f, +2.f, +2.f, +1.f,
    };
    const float y_train[] = {
        0.f, 0.f, 0.f, 1.f, 1.f, 1.f,
    };

    const auto x_train_table =
        dal::homogen_table{ row_count_train, column_count, x_train };
    const auto y_train_table =
        dal::homogen_table{ row_count_train, 1, y_train };

    const auto kernel_desc =
        dal::linear_kernel::descriptor{}.set_k(1.0).set_b(0.0);

    const auto svm_desc =
        dal::svm::descriptor<float,
                             dal::svm::task::classification,
                             dal::svm::method::smo>{ kernel_desc }
            .set_c(1.0)
            .set_accuracy_threshold(0.01)
            .set_max_iteration_count(100)
            .set_cache_size(200.0)
            .set_shrinking(false)
            .set_tau(1e-6);

    const auto result_train =
        dal::train(svm_desc, x_train_table, y_train_table);

    std::cout << "Bias:" << std::endl << result_train.get_bias() << std::endl;
    std::cout << "Support indices:" << std::endl
              << result_train.get_support_indices() << std::endl;

    constexpr std::int64_t row_count_test = 3;
    const float x_test[] = {
        -1.f, -1.f, +2.f, +2.f, +3.f, +2.f,
    };
    const float y_true[] = {
        0.f,
        1.f,
        1.f,
    };

    const auto x_test_table =
        dal::homogen_table{ row_count_test, column_count, x_test };
    const auto y_true_table = dal::homogen_table{ row_count_test, 1, y_true };

    const auto result_test =
        dal::infer(svm_desc, result_train.get_model(), x_test_table);

    std::cout << "Decision function result:" << std::endl
              << result_test.get_decision_function() << std::endl;
    std::cout << "Labels result:" << std::endl
              << result_test.get_labels() << std::endl;
    std::cout << "Labels true:" << std::endl << y_true_table << std::endl;

    return 0;
}
