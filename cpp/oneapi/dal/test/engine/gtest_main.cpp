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

#include <gtest/gtest.h>
#include "oneapi/dal/test/engine/config.hpp"

int main(int argc, char** argv) {
    using oneapi::dal::test::engine::global_config;
    oneapi::dal::test::engine::global_setup(global_config{});

    ::testing::InitGoogleTest(&argc, argv);
    const int status = RUN_ALL_TESTS();

    oneapi::dal::test::engine::global_cleanup();
    return status;
}
