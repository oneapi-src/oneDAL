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

#include "oneapi/dal/detail/ops_dispatcher.hpp"

namespace oneapi::dal::detail {
namespace v1 {

template <typename Descriptor, typename Tag>
struct train_ops;

template <typename Descriptor>
using tagged_train_ops = train_ops<Descriptor, typename Descriptor::tag_t>;

template <typename Head, typename... Tail>
auto train_dispatch(Head&& head, Tail&&... tail) {
    using dispatcher_t = ops_policy_dispatcher<std::decay_t<Head>, tagged_train_ops>;
    return dispatcher_t{}(std::forward<Head>(head), std::forward<Tail>(tail)...);
}

} // namespace v1

using v1::train_dispatch;

} // namespace oneapi::dal::detail
