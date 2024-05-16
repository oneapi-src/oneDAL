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

#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

namespace oneapi::dal::linear_regression::backend {

#ifdef ONEDAL_DATA_PARALLEL

using alloc = sycl::usm::alloc;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

/// Adds ridge penalty to the diagonal elements of the xtx matrix

///
/// @tparam Float Floating-point type used to perform computations
///
/// @param[in]  q                 The SYCL queue
/// @param[in]  xtx               The input matrix to which the ridge penalty is added
/// @param[in]  compute_intercept Flag indicating whether the intercept term is used in the matrix
/// @param[in]  alpha             The regularization parameter
/// @param[in]  deps              Events indicating the availability of the `xtx` for reading or writing
///
/// @return A SYCL event indicating the availability of the matrix for reading and writing
template <typename Float>
sycl::event add_ridge_penalty(sycl::queue& q,
                              const pr::ndarray<Float, 2>& xtx,
                              bool compute_intercept,
                              Float alpha,
                              const bk::event_vector& deps = {}) {
    ONEDAL_ASSERT(xtx.has_mutable_data());
    ONEDAL_ASSERT(be::is_known_usm(q, xtx.get_mutable_data()));
    ONEDAL_ASSERT(xtx.get_dimension(0) == xtx.get_dimension(1));

    Float* xtx_ptr = xtx.get_mutable_data();
    std::int64_t feature_count = xtx.get_dimension(0);
    std::int64_t original_feature_count = feature_count - compute_intercept;

    return q.submit([&](sycl::handler& cgh) {
        const auto range = be::make_range_1d(original_feature_count);
        cgh.depends_on(deps);
        std::int64_t step = feature_count + 1;
        cgh.parallel_for(range, [=](sycl::id<1> idx) {
            xtx_ptr[idx * step] += alpha;
        });
    });
}

} // namespace oneapi::dal::linear_regression::backend

#endif // ONEDAL_DATA_PARALLEL
