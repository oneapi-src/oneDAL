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

#include <unordered_map>
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::test::engine {

#define INSTANTIATE_TYPE_MAP(T)       \
    const char* type2str<T>::name() { \
        return #T;                    \
    }

INSTANTIATE_TYPE_MAP(float)
INSTANTIATE_TYPE_MAP(double)
INSTANTIATE_TYPE_MAP(std::uint8_t)
INSTANTIATE_TYPE_MAP(std::uint16_t)
INSTANTIATE_TYPE_MAP(std::uint32_t)
INSTANTIATE_TYPE_MAP(std::uint64_t)
INSTANTIATE_TYPE_MAP(std::int8_t)
INSTANTIATE_TYPE_MAP(std::int16_t)
INSTANTIATE_TYPE_MAP(std::int32_t)
INSTANTIATE_TYPE_MAP(std::int64_t)

std::unordered_map<std::string, global_setup_action*> global_actions;

void register_global_setup(const std::string& name, global_setup_action* action) {
    const auto it = global_actions.find(name);
    if (it != global_actions.end()) {
        throw std::runtime_error{ "Global setup action `" + name + "` is already registered" };
    }
    global_actions[name] = action;
}

void init_global_setup_actions(const global_config& config) {
    for (const auto& [name, action] : global_actions) {
        action->init(config);
    }
}

void tear_down_global_setup_actions() {
    for (const auto& [name, action] : global_actions) {
        action->tear_down();
        delete action;
    }
}

#ifdef ONEDAL_DATA_PARALLEL
test_queue_provider& test_queue_provider::get_instance() {
    static test_queue_provider provider;
    return provider;
}

bool device_test_policy::has_native_float64() const {
#ifdef ONEDAL_DISABLE_FP64_TESTS
    return false;
#else
    const auto device = queue_.get_device();
    const auto fp_config = device.get_info<sycl::info::device::double_fp_config>();
    const bool float64_support = !fp_config.empty();

    return float64_support;
#endif
}
#endif

#ifdef ONEDAL_DATA_PARALLEL
class test_queue_provider_setup : public global_setup_action {
public:
    void init(const global_config& config) override {
        const auto queue = get_queue(config.device_selector);
        test_queue_provider::get_instance().init(queue);
    }

    void tear_down() override {
        test_queue_provider::get_instance().reset();
    }

private:
    static sycl::queue get_default_queue() {
        try {
            return sycl::queue{ sycl::gpu_selector_v };
        }
        catch (const std::exception& ex) {
            return sycl::queue{ sycl::cpu_selector_v };
        }
    }

    static sycl::queue get_queue(const std::string& device_selector) {
        if (device_selector.empty()) {
            return get_default_queue();
        }

        if (device_selector == "cpu") {
            return sycl::queue{ sycl::cpu_selector_v };
        }

        if (device_selector == "gpu") {
            return sycl::queue{ sycl::gpu_selector_v };
        }

        throw std::invalid_argument{ "Unknown device selector" };
    }
};
#else
class test_queue_provider_setup : public global_setup_action {
public:
    void init(const global_config& config) override {
        if (config.device_selector != "" && config.device_selector != "cpu") {
            throw std::invalid_argument{
                "Test is build in HOST mode, so only CPU device is available"
            };
        }
    }

    void tear_down() override {}
};
#endif

REGISTER_GLOBAL_SETUP(test_queue_provider, test_queue_provider_setup)

} // namespace oneapi::dal::test::engine
