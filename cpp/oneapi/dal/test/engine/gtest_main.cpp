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

#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>

#include <gtest/gtest.h>
#include "oneapi/dal/test/engine/config.hpp"

using oneapi::dal::test::engine::global_config;

static std::vector<std::string> split_string(const std::string& entry, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(entry);

    std::string item;
    while (std::getline(ss, item, delim)) {
        result.push_back(item);
    }

    return result;
}

static std::tuple<std::string, std::string> split_key_value(const std::string& entry) {
    const auto key_value = split_string(entry, '=');
    if (key_value.size() > 2) {
        throw std::invalid_argument{ "Invalid format of input argument: '" + entry + "'" };
    }
    return { key_value[0], key_value[1] };
}

template <typename Contrainer, typename Setter>
static void try_add_config_key(Contrainer& options, const std::string& key, Setter&& setter) {
    const auto it = options.find("--" + key);
    if (it != options.end()) {
        setter(it->second);
        options.erase(it);
    }
}

static global_config parse_config(int argc, char** argv) {
    std::unordered_map<std::string, std::string> options;
    for (int i = 1; i < argc; i++) {
        const auto entry = std::string{ argv[i] };
        const auto [key, value] = split_key_value(entry);
        options[key] = value;
    }

    global_config config;
    try_add_config_key(options, "device", [&](const std::string& x) {
        config.device_selector = x;
    });

    if (!options.empty()) {
        throw std::invalid_argument{ "Found unknown options" };
    }

    return config;
}

int main(int argc, char** argv) {
    const auto config = parse_config(argc, argv);
    oneapi::dal::test::engine::global_setup(config);

    ::testing::InitGoogleTest(&argc, argv);
    const int status = RUN_ALL_TESTS();

    oneapi::dal::test::engine::global_cleanup();
    return status;
}
