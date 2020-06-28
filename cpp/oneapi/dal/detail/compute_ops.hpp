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

#include "oneapi/dal/execution_context.hpp"

namespace oneapi::dal::detail {

template <typename Descriptor, typename Tag>
struct compute_ops;

template <typename Context, typename Descriptor, typename Head, typename... Tail>
auto compute_dispatch_by_input(const Context& ctx,
                               const Descriptor& desc,
                               Head&& head,
                               Tail&&... tail) {
    using tag_t   = typename Descriptor::tag_t;
    using ops_t   = compute_ops<Descriptor, tag_t>;
    using input_t = typename ops_t::input_t;

    if constexpr (std::is_same_v<std::decay_t<Head>, input_t>) {
        return ops_t()(ctx, desc, std::forward<Head>(head), std::forward<Tail>(tail)...);
    }

    const auto input = input_t{ std::forward<Head>(head), std::forward<Tail>(tail)... };
    return ops_t()(ctx, desc, input);
}

template <typename Head, typename... Tail>
auto compute_dispatch_by_ctx(Head&& head, Tail&&... tail) {
    using tag_t = typename std::decay_t<Head>::tag_t;
    if constexpr (std::is_same_v<tag_t, detail::execution_context_tag>) {
        return compute_dispatch_by_input(head, std::forward<Tail>(tail)...);
    }

    return compute_dispatch_by_input(default_execution_context(),
                                     std::forward<Head>(head),
                                     std::forward<Tail>(tail)...);
}

} // namespace oneapi::dal::detail
