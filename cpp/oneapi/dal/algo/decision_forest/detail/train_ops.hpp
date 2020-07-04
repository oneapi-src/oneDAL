/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/algo/decision_forest/train_types.hpp"

namespace oneapi::dal::decision_forest::detail {

template <typename Context, typename Float, typename Task, typename Method>
struct train_ops_dispatcher {
    train_result<Task> operator()(const Context&,
                                  const descriptor_base<Task>&,
                                  const train_input<Task>&) const;
};

template <typename Descriptor>
struct train_ops {
    using float_t           = typename Descriptor::float_t;
    using task_t            = typename Descriptor::task_t;
    using method_t          = typename Descriptor::method_t;
    using input_t           = train_input<task_t>;
    using result_t          = train_result<task_t>;
    using descriptor_base_t = descriptor_base<task_t>;

    void validate(const Descriptor& params, const input_t& input) const {}

    template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const input_t& input) const {
        validate(desc, input);
        return train_ops_dispatcher<Context, float_t, task_t, method_t>()(ctx, desc, input);
    }
};

} // namespace oneapi::dal::decision_forest::detail
