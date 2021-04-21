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

#include "oneapi/dal/test/engine/config.hpp"

#include <stdexcept>
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::test::engine {

#ifdef ONEDAL_DATA_PARALLEL

static sycl::queue get_default_queue() {
    try {
        return sycl::queue{ sycl::gpu_selector{} };
    }
    catch (const sycl::runtime_error& ex) {
        return sycl::queue{ sycl::cpu_selector{} };
    }
}

static sycl::queue get_queue(const std::string& device_selector) {
    if (device_selector.empty()) {
        return get_default_queue();
    }

    if (device_selector == "cpu") {
        return sycl::queue{ sycl::cpu_selector{} };
    }

    if (device_selector == "gpu") {
        return sycl::queue{ sycl::gpu_selector{} };
    }

    if (device_selector == "host") {
        return sycl::queue{ sycl::host_selector{} };
    }

    throw std::invalid_argument{ "Unknown device selector" };
}

void global_setup(const global_config& config) {
    test_queue_provider::get_instance().init(get_queue(config.device_selector));
}

void global_cleanup() {
    test_queue_provider::get_instance().reset();
}

#else
void global_setup(const global_config& config) {
    if (config.device_selector != "" && config.device_selector != "cpu") {
        throw std::invalid_argument{
            "Test is build in HOST mode, so only CPU device is available"
        };
    }
}

void global_cleanup() {}
#endif

} //namespace oneapi::dal::test::engine
