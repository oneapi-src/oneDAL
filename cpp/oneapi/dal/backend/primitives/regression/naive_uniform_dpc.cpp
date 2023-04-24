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

#include <algorithm>

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"

#include "oneapi/dal/backend/primitives/regression/uniform.hpp"

namespace oneapi::dal::backend::primitives {

template <typename ResponseType>
naive_uniform_regression<ResponseType>::naive_uniform_regression(sycl::queue& q) : base_t{ q } {}

template <typename ResponseType>
sycl::event normalize(sycl::queue& queue,
                      std::int64_t n_responses,
                      ndview<ResponseType, 1>& results,
                      const event_vector& deps) {
    ONEDAL_ASSERT(results.has_mutable_data());
    const ResponseType factor = 1.0 / double(n_responses);
    const auto range = make_range_1d(results.get_dimension(0));
    auto* res_ptr = results.get_mutable_data();
    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<1> idx) {
            res_ptr[idx] *= factor;
        });
    });
}

template <typename ResponseType>
sycl::event naive_uniform_regression<ResponseType>::operator()(
    const ndview<ResponseType, 2>& responses,
    ndview<ResponseType, 1>& results,
    const event_vector& deps) {
    constexpr sum<ResponseType> binary_op{};
    constexpr identity<ResponseType> unary_op{};
    ONEDAL_PROFILER_TASK(regression.uniform, this->get_queue());
    ONEDAL_ASSERT(results.get_dimension(0) == responses.get_dimension(0));
    auto reduction_event =
        reduce_by_rows(this->get_queue(), responses, results, binary_op, unary_op, deps);
    const auto n_responses = responses.get_dimension(1);
    return normalize(this->get_queue(), n_responses, results, { reduction_event });
}

#define INSTANTIATE(RESPONSE) template class naive_uniform_regression<RESPONSE>;

INSTANTIATE(float);
INSTANTIATE(double);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
