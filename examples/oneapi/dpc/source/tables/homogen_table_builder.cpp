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

#define ONEAPI_DAL_DATA_PARALLEL
#include "oneapi/dal/data/accessor.hpp"

#include "example_util/utils.hpp"

using namespace oneapi;

void run(sycl::queue& queue) {
    constexpr std::int64_t row_count = 5;
    constexpr std::int64_t column_count = 3;

    dal::homogen_table_builder builder;
    builder.set_data_type(dal::data_type::float32)
           .allocate(queue, row_count, column_count);

    {
        dal::row_accessor<float> acc { builder };
        auto rows = acc.pull(queue);

        auto event = queue.submit([&](sycl::handler& cgh) {
            rows.need_mutable_data(queue);
            auto data = rows.get_mutable_data();
            cgh.parallel_for(sycl::range<1>(rows.get_count()), [=](sycl::id<1> idx) {
                data[idx[0]] = float(idx);
            });
        });
        event.wait();

        acc.push(queue, rows);
    }

    auto table = builder.build();
    std::cout << "Table values:" << std::endl
              << table << std::endl;
}

int main(int argc, char const *argv[]) {
    for (auto device : list_devices()) {
        std::cout << "Running on "
                  << device.get_info<sycl::info::device::name>()
                  << std::endl << std::endl;
        auto queue = sycl::queue{device};
        run(queue);
    }
    return 0;
}
