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
#include <iostream>
#include <unordered_map>

#include "oneapi/dal/test/engine/catch.hpp"
#include "oneapi/dal/test/engine/config.hpp"

inline constexpr int default_benchmark_run_count = 5;

int main(int argc, char** argv) {
    using namespace Catch::Clara;
    using oneapi::dal::test::engine::global_config;

    global_config config;
    Catch::Session session;

    auto cli =
        session.cli() | Opt(config.device_selector, "device")["--device"]("DPC++ device selector");

    session.cli(cli);
    session.configData().benchmarkSamples = default_benchmark_run_count;

    const int parse_status = session.applyCommandLine(argc, argv);
    if (parse_status != 0) {
        std::cerr << "Command line arguments parsing error" << std::endl;
        return parse_status;
    }

    oneapi::dal::test::engine::global_setup(config);
    const int status = session.run();
    oneapi::dal::test::engine::global_cleanup();
    return status;
}
