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

#include <iomanip>
#include <iostream>
#include <CL/sycl.hpp>

#define ONEAPI_DAL_DATA_PARALLEL
#include "oneapi/dal/algo/pca.hpp"
#include "oneapi/dal/data/accessor.hpp"

using namespace oneapi;

std::ostream &operator <<(std::ostream& stream, const dal::table& table) {
    auto arr = dal::row_accessor<const float>(table).pull();
    const auto x = arr.get_data();

    for (std::int64_t i = 0; i < table.get_row_count(); i++) {
        for (std::int64_t j = 0; j < table.get_column_count(); j++) {
            std::cout << std::setw(10)
                      << std::setiosflags(std::ios::fixed)
                      << std::setprecision(3)
                      << x[i * table.get_column_count() + j];
        }
        std::cout << std::endl;
    }
    return stream;
}

template <typename Selector>
void try_add_device(std::vector<sycl::device>& devices) {
    try {
        devices.push_back(Selector{}.select_device());
    }
    catch (...) {}
}

std::vector<sycl::device> list_devices() {
    std::vector<sycl::device> devices;
    try_add_device<sycl::host_selector>(devices);
    try_add_device<sycl::cpu_selector>(devices);
    try_add_device<sycl::gpu_selector>(devices);
    return devices;
}

void run(sycl::queue& queue) {
    constexpr std::int64_t row_count = 5;
    constexpr std::int64_t column_count = 3;

    const float data_host[] = {
        1.f,  2.f,  3.f,
        1.f,  -1.f, 0.f,
        4.f,  5.f,  6.f,
        1.f,  2.f,  5.f,
        -4.f, 3.f,  0.f
    };
    auto data = sycl::malloc_shared<float>(row_count * column_count, queue);
    queue.memcpy(data, data_host, sizeof(float) * row_count * column_count).wait();

    const auto data_table = dal::homogen_table{ row_count, column_count, data };

    const auto pca_desc = dal::pca::descriptor<>()
        .set_component_count(3)
        .set_is_deterministic(true);

    const auto result = dal::train(queue, pca_desc, data_table);

    std::cout << "Eigenvectors:" << std::endl
              << result.get_eigenvectors() << std::endl;

    std::cout << "Eigenvalues:" << std::endl
              << result.get_eigenvalues() << std::endl;

    sycl::free(data, queue);
}

int main(int argc, char const *argv[]) {
    for (auto device : list_devices()) {
        auto queue = sycl::queue{device};
        run(queue);
    }
    return 0;
}
