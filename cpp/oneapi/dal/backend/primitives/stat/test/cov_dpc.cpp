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
#include "oneapi/dal/backend/primitives/stat/cov.hpp"

namespace oneapi::dal::backend::primitives::test {

TEST("cov", "[cov]") {
    DECLARE_TEST_POLICY(policy);
    auto& queue = policy.get_queue();

    auto [sums, sums_event] = ndarray<float, 1>::full(queue, { 100 }, 1000.f, sycl::usm::alloc::device);
    auto [data, data_event] = ndarray<float, 2>::ones(queue, { 1000, 100 }, sycl::usm::alloc::device);
    auto cov = ndarray<float, 2>::empty(queue, { 100, 100 });

    compute_cov<float>(queue, data, sums, cov, { sums_event, data_event }).wait_and_throw();
}

} // namespace oneapi::dal::backend::primitives::test
