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
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/backend/primitives/stat/cov.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

TEST("400K x 1K", "[cor][perf]") {
    DECLARE_TEST_POLICY(policy);
    SKIP_IF(policy.is_cpu());

    auto& queue = policy.get_queue();
    const auto alloc = sycl::usm::alloc::shared;

    // 4 x 400K x 1K ~ 1.526Gb
    const auto df = GENERATE_DATAFRAME(te::dataframe_builder{ 400000, 1000 }.fill_uniform(-1, 1));

    const auto column_count = df.get_column_count();
    auto corr = ndarray<float_t, 2>::empty(queue, { column_count, column_count }, alloc);
    auto means = ndarray<float_t, 1>::empty(queue, { column_count }, alloc);
    auto vars = ndarray<float_t, 1>::empty(queue, { column_count }, alloc);
    auto tmp = ndarray<float_t, 1>::empty(queue, { column_count }, alloc);

    // This is not valid sums across `df`, but it's OK for benchmarking
    ndarray<float, 1> sums;
    sycl::event sums_event;
    std::tie(sums, sums_event) = ndarray<float_t, 1>::zeros(queue, { column_count }, alloc);

    // Table is allocated using shared USM
    const auto data = df.get_table(policy, te::table_id::homogen<float_t>());

    // We need to wait until all previously submitted kernels are executed
    queue.wait_and_throw();

    BENCHMARK("correlation") {
        correlation(queue, data, sums, corr, means, vars, tmp, { sums_event }).wait_and_throw();
    };
}

} // namespace oneapi::dal::backend::primitives::test
