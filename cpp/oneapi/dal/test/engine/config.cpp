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
#include "oneapi/dal/test/engine/common.hpp"

namespace oneapi::dal::test::engine {

void global_setup(const global_config& config) {
    init_global_setup_actions(config);
}

void global_cleanup() {
    tear_down_global_setup_actions();
}

} //namespace oneapi::dal::test::engine
