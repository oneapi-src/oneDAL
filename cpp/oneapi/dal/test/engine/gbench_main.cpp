#include <benchmark/benchmark.h>
#include "oneapi/dal/test/engine/config.hpp"
#include <iostream>

//BENCHMARK_MAIN();

//initialize queue/device selector/queue provider?? whatever here?

int main(int argc, char** argv) {
    using oneapi::dal::test::engine::global_config;

    global_config config;

    config.device_selector = "cpu";

    oneapi::dal::test::engine::global_setup(config);

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    oneapi::dal::test::engine::global_cleanup();

    return 0;
}