#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

#include "oneapi/dal/test/engine/config.hpp"

int main(int argc, char** argv) {
    using namespace Catch::clara;
    using oneapi::dal::test::engine::global_config;

    global_config config;
    Catch::Session session;

    auto cli = session.cli() |
        Opt(config.device_selector, "device")
        ["--device"]
        ("DPC++ device selector");

    session.cli(cli);

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
