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

#define ONEAPI_DAL_DATA_PARALLEL
#include "oneapi/dal/algo/knn.hpp"
#include "oneapi/dal/data/table.hpp"

#include "oneapi/dal/exceptions.hpp"
#include "example_util/utils.hpp"

using namespace oneapi;

void run(sycl::queue& queue) {
    constexpr std::int64_t row_count = 5;
    constexpr std::int64_t column_count = 3;

    const float x_train_host[] = {1.f, 2.f, 3.f, 1.f, -1.f, 0.f, 4.f, 5.f,
                            6.f, 1.f, 2.f, 5.f, -4.f, 3.f, 0.f};

    const float y_train_host[] = {0, 1, 0, 1, 1};

    auto x_train = sycl::malloc_shared<float>(row_count * column_count, queue);
    queue.memcpy(x_train, x_train_host, sizeof(float) * row_count * column_count).wait();
    const auto x_train_table = dal::homogen_table{ row_count, column_count, x_train };

    auto y_train = sycl::malloc_shared<float>(row_count * 1, queue);
    queue.memcpy(y_train, y_train_host, sizeof(float) * row_count * 1).wait();
    const auto y_train_table = dal::homogen_table{ row_count, 1, y_train };

    const auto knn_desc =
        dal::knn::descriptor<float, oneapi::dal::knn::method::brute_force>()
            .set_class_count(2)
            .set_neighbor_count(1)
            .set_data_use_in_model(false);

    const float x_test_host[] = {1.f, 2.f, 2.f, 1.f, -1.f, 1.f, 4.f, 6.f,
                            6.f, 2.f, 2.f, 5.f, -4.f, 3.f, 1.f};

    const float y_test_host[] = {0, 1, 0, 1, 1};

    auto x_test = sycl::malloc_shared<float>(row_count * column_count, queue);
    queue.memcpy(x_test , x_test_host, sizeof(float) * row_count * column_count).wait();
    const auto x_test_table = dal::homogen_table{ row_count, column_count, x_test };

    auto y_test = sycl::malloc_shared<float>(row_count * 1, queue);
    queue.memcpy(y_test, y_test_host, sizeof(float) * row_count * 1).wait();
    const auto y_test_table = dal::homogen_table{ row_count, 1, y_test };

    try {
        const auto train_result = dal::train(queue, knn_desc, x_train_table, y_train_table);

        const auto test_result =
            dal::infer(queue, knn_desc, x_test_table, train_result.get_model());

        std::cout << "Test results:" << std::endl
                << test_result.get_labels() << std::endl;
        std::cout << "True labels:" << std::endl << y_test_table << std::endl;
    }
    catch(oneapi::dal::unimplemented_error& e) {
        std::cout << "  " << e.what() << std::endl;
        return;
    }
    sycl::free(x_train, queue);
    sycl::free(y_train, queue);
    sycl::free(x_test, queue);
    sycl::free(y_test, queue);
}

int main(int argc, char const *argv[]) {
    for (auto device : list_devices()) {
        std::cout << "Running on "
              << device.get_info<sycl::info::device::name>()
              << std::endl;
        auto queue = sycl::queue{device};
        run(queue);
    }
    return 0;
}
