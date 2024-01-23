/*******************************************************************************
* Copyright 2024 Intel Corporation
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
#include "oneapi/dal/detail/cpu_info.hpp"

namespace oneapi::dal::test {

TEST("can create default CPU info") {
    const detail::cpu_info default_cpu_info;

    REQUIRE(detail::cpu_vendor::intel == default_cpu_info.get_cpu_vendor());
}

} // namespace oneapi::dal::test