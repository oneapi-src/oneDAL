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

#include "oneapi/dal/policy.hpp"

namespace oneapi::dal::detail {

template <typename Descriptor, typename Tag>
struct train_ops;

template <typename Policy, typename Descriptor, typename Head, typename... Tail>
auto train_dispatch_by_input(const Policy& policy,
                             const Descriptor& desc,
                             Head&& head,
                             Tail&&... tail) {
    using tag_t   = typename Descriptor::tag_t;
    using ops_t   = train_ops<Descriptor, tag_t>;
    using input_t = typename ops_t::input_t;

    if constexpr (std::is_same_v<std::decay_t<Head>, input_t>) {
        return ops_t{}(policy, desc, std::forward<Head>(head), std::forward<Tail>(tail)...);
    }
    else {
        const auto input = input_t{ std::forward<Head>(head), std::forward<Tail>(tail)... };
        return ops_t{}(policy, desc, input);
    }
}

template <typename Head, typename... Tail>
auto train_dispatch_by_policy(Head&& head, Tail&&... tail) {
    if constexpr (is_execution_policy_v<std::decay_t<Head>>) {
        return train_dispatch_by_input(std::forward<Head>(head), std::forward<Tail>(tail)...);
    }
    else {
        return train_dispatch_by_input(default_policy{},
                                       std::forward<Head>(head),
                                       std::forward<Tail>(tail)...);
    }
}

} // namespace oneapi::dal::detail
