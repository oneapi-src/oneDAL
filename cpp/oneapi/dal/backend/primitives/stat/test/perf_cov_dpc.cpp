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
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/backend/primitives/stat/cov.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace pr = dal::backend::primitives;

TEST("100K x 4K", "[cor][perf]") {
    DECLARE_TEST_POLICY(policy);
    SKIP_IF(policy.is_cpu());

    auto& q = policy.get_queue();
    const std::int64_t row_count = 100000;
    const std::int64_t column_count = 4000;
    const auto alloc = sycl::usm::alloc::device;

    // 4 x 400K x 1K ~ 1.526Gb
    const auto data_mat =
        la::generate_uniform_matrix<float_t>({ row_count, column_count }, -2.0, 5.0, 7777);
    const auto data =
        ndarray<float_t, 2>::wrap(data_mat.to_device(q).get_array(), { row_count, column_count });

    auto corr = ndarray<float_t, 2>::empty(q, { column_count, column_count }, alloc);
    auto means = ndarray<float_t, 1>::empty(q, { column_count }, alloc);
    auto vars = ndarray<float_t, 1>::empty(q, { column_count }, alloc);
    auto tmp = ndarray<float_t, 1>::empty(q, { column_count }, alloc);

    // This is not valid sums across `df`, but it's OK for benchmarking
    ndarray<float, 1> sums;
    sycl::event sums_event;
    std::tie(sums, sums_event) = ndarray<float_t, 1>::zeros(q, { column_count }, alloc);

    // We need to wait until all previously submitted kernels are executed
    q.wait_and_throw();
    auto gemm_event = pr::gemm(q, data.t(), data, corr, float_t(1), float_t(0));
    gemm_event.wait_and_throw();
    BENCHMARK("correlation") {
        correlation(q, data.get_dimension(0), sums, corr, tmp, { sums_event }).wait_and_throw();
    };
}

} // namespace oneapi::dal::backend::primitives::test
