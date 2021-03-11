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

TEST("just fill", "[dpc++]") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    float* x = sycl::malloc_shared<float>(10000, q);
    q.fill(x, -1.0f, 10000).wait_and_throw();

    sycl::free(x, q);
}

TEST_CASE("fill and write on host", "[dpc++]") {
    DECLARE_TEST_POLICY(policy);
    auto& q = policy.get_queue();

    float* x = sycl::malloc_shared<float>(10000, q);
    q.fill(x, -1.0f, 10000).wait_and_throw();

    x[0] = 0.0f;

    sycl::free(x, q);
}

} // namespace oneapi::dal::backend::primitives::test
