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

namespace oneapi::dal::backend::primitives::test {

TEST("fill device USM and copy to host", "[usm]") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    const std::int64_t element_count = 10;
    const float filler = GENERATE(1.5f, 0.25f, -1.75f);

    float* data = sycl::malloc_device<float>(element_count, q);
    q.fill(data, (float)filler, element_count).wait_and_throw();

    std::vector<float> data_host_vec(element_count);
    float* data_host = data_host_vec.data();
    INFO("copy to host");
    q.memcpy(data_host, data, sizeof(float) * element_count).wait_and_throw();

    SECTION("check filler on host") {
        for (std::int64_t i = 0; i < element_count; i++) {
            const float x = data_host[i];
            CAPTURE(i);
            REQUIRE(x == filler);
        }
    }

    sycl::free(data, q);
}

} // namespace oneapi::dal::backend::primitives::test
