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

#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::test::engine {

test_queue_provider& test_queue_provider::get_instance() {
    static test_queue_provider provider;
    return provider;
}

static bool check_if_env_knob_is_enabled(const char* env_var) {
    const char* var = std::getenv(env_var);
    if (!var) {
        return false;
    }

    try {
        return std::stoi(var) > 0;
    }
    catch (std::invalid_argument&) {
        return false;
    }
}

static bool check_if_env_overrides_fp64_settings() {
    return check_if_env_knob_is_enabled("OverrideDefaultFP64Settings");
}

static bool check_if_env_forces_dp_emulation() {
    return check_if_env_knob_is_enabled("IGC_EnableDPEmulation") ||
           check_if_env_knob_is_enabled("IGC_ForceDPEmulation");
}

bool device_test_policy::has_native_float64() const {
    const auto device = queue_.get_device();
    const auto fp_config = device.get_info<sycl::info::device::double_fp_config>();
    const bool float64_support = !fp_config.empty();
    const bool emulated = check_if_env_overrides_fp64_settings() && //
                          check_if_env_forces_dp_emulation();
    return float64_support && !emulated;
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

} // namespace oneapi::dal::test::engine
