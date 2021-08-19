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

#include <benchmark/benchmark.h>
#include "oneapi/dal/test/engine/config.hpp"
#include <iostream>

int main(int argc, char** argv) {
    using oneapi::dal::test::engine::global_config;

    global_config config;
#ifdef ONEDAL_DATA_PARALLEL
    config.device_selector = "gpu";
#else
    config.device_selector = "cpu";
#endif

    oneapi::dal::test::engine::global_setup(config);

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    oneapi::dal::test::engine::global_cleanup();

    return 0;
}
