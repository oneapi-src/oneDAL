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

#include <CL/sycl.hpp>

#define ONEAPI_DAL_DATA_PARALLEL
#include "oneapi/dal/algo/linear_kernel.hpp"

#include "example_util/utils.hpp"

using namespace oneapi;

void run(sycl::queue &queue) {
    std::cout << "Running on " << queue.get_device().get_info<sycl::info::device::name>()
              << std::endl
              << std::endl;

    constexpr std::int64_t row_count_x  = 2;
    constexpr std::int64_t row_count_y  = 3;
    constexpr std::int64_t column_count = 3;

    const float x_host[] = {
        1.f, 2.f, 3.f, 1.f, -1.f, 0.f,
    };

    const float y_host[] = {
        1.f, 2.f, 3.f, 1.f, -1.f, 0.f, 4.f, 5.f, 6.f,
    };

    auto x = sycl::malloc_shared<float>(row_count_x * column_count, queue);
    queue.memcpy(x, x_host, sizeof(float) * row_count_x * column_count).wait();

    auto y = sycl::malloc_shared<float>(row_count_y * column_count, queue);
    queue.memcpy(y, y_host, sizeof(float) * row_count_y * column_count).wait();

    const auto x_table     = dal::homogen_table{ queue, row_count_x, column_count, x };
    const auto y_table     = dal::homogen_table{ queue, row_count_y, column_count, y };
    const auto kernel_desc = dal::linear_kernel::descriptor{}.set_scale(2.0).set_shift(1.0);

    const auto result = dal::compute(queue, kernel_desc, x_table, y_table);

    std::cout << "Values:" << std::endl << result.get_values() << std::endl;

    sycl::free(x, queue);
    sycl::free(y, queue);
}

int main(int argc, char const *argv[]) {
    for (auto device : list_devices()) {
        std::cout << "Running on " << device.get_info<sycl::info::device::name>() << std::endl
                  << std::endl;
        auto queue = sycl::queue{ device };
        run(queue);
    }
    return 0;
}
