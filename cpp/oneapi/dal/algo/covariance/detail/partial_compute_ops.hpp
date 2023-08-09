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

#pragma once

#include "oneapi/dal/algo/covariance/compute_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::covariance::detail {
namespace v1 {
//TODO: add checks and refactor this file
template <typename Context, typename Float, typename Method, typename Task, typename... Options>
struct partial_compute_ops_dispatcher {
    compute_result<Task> operator()(const Context&,
                                    const descriptor_base<Task>&,
                                    const partial_compute_input<Task>&) const;
};

template <typename Descriptor>
struct partial_compute_ops {
    using float_t = typename Descriptor::float_t;
    using method_t = typename Descriptor::method_t;
    using task_t = typename Descriptor::task_t;
    using input_t = partial_compute_input<task_t>;
    using result_t = compute_result<task_t>;
    using descriptor_base_t = descriptor_base<task_t>;

    template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const input_t& input) const {
        const auto result =
            partial_compute_ops_dispatcher<Context, float_t, method_t, task_t>()(ctx, desc, input);
        return result;
    }
};

} // namespace v1

using v1::partial_compute_ops;

} // namespace oneapi::dal::covariance::detail
