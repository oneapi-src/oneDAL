/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/spmd/common.hpp"

namespace oneapi::dal::detail {
namespace v1 {

struct ops_error_handling_dispatcher {
    template <typename Policy, typename Ops>
    auto operator()(Policy&& policy, Ops&& op) {
        using result_t = decltype(op());
        if constexpr (is_distributed_policy_v<std::decay_t<Policy>>) {
            try {
                try {
                    return op();
                }
                catch (const dal::preview::spmd::error_holder& e) {
                    throw e;
                }
                catch (...) {
                    policy.get_communicator().set_active_exception(std::current_exception());
                }
                policy.get_communicator().wait_for_exception_handling();
            }
            catch (const dal::preview::spmd::error_holder& e) {
                e.rethrow_actual();
            }
        }
        else {
            return op();
        }

        return result_t{};
    }
};

template <typename T, typename Ops, bool IsInput = std::is_same_v<T, typename Ops::input_t>>
struct ops_input_dispatcher;

template <typename T, typename Ops>
struct ops_input_dispatcher<T, Ops, /* IsInput = */ true> {
    template <typename... Args>
    auto operator()(Args&&... args) {
        return Ops{}(std::forward<Args>(args)...);
    }
};

template <typename T, typename Ops>
struct ops_input_dispatcher<T, Ops, /* IsInput = */ false> {
    template <typename Policy, typename Descriptor, typename... Args>
    auto operator()(Policy&& policy, Descriptor&& desc, Args&&... args) {
        using input_t = typename Ops::input_t;

        return ops_error_handling_dispatcher{}(std::forward<Policy>(policy), [&]() {
            return Ops{}(std::forward<Policy>(policy),
                         std::forward<Descriptor>(desc),
                         input_t{ std::forward<Args>(args)... });
        });
    }
};

template <typename T, template <typename> typename Ops, bool IsPolicy = is_execution_policy_v<T>>
struct ops_policy_dispatcher;

template <typename T, template <typename> typename Ops>
struct ops_policy_dispatcher<T, Ops, /* IsPolicy = */ true> {
    template <typename Policy, typename Descriptor, typename Head, typename... Tail>
    auto operator()(Policy&& policy, Descriptor&& desc, Head&& head, Tail&&... tail) {
        using ops_t = Ops<std::decay_t<Descriptor>>;
        using dispatcher_t = ops_input_dispatcher<std::decay_t<Head>, ops_t>;
        return dispatcher_t{}(std::forward<Policy>(policy),
                              std::forward<Descriptor>(desc),
                              std::forward<Head>(head),
                              std::forward<Tail>(tail)...);
    }
};

template <typename T, template <typename> typename Ops>
struct ops_policy_dispatcher<T, Ops, /* IsPolicy = */ false> {
    template <typename Descriptor, typename Head, typename... Tail>
    auto operator()(Descriptor&& desc, Head&& head, Tail&&... tail) {
        using ops_t = Ops<std::decay_t<Descriptor>>;
        using dispatcher_t = ops_input_dispatcher<std::decay_t<Head>, ops_t>;
        return dispatcher_t{}(host_policy::get_default(),
                              std::forward<Descriptor>(desc),
                              std::forward<Head>(head),
                              std::forward<Tail>(tail)...);
    }
};

template <typename Object,
          typename T,
          template <typename, typename>
          typename Ops,
          bool IsPolicy = is_execution_policy_v<T>>
struct ops_policy_dispatcher_object;

template <typename Object, typename T, template <typename, typename> typename Ops>
struct ops_policy_dispatcher_object<Object, T, Ops, /* IsPolicy = */ true> {
    template <typename Policy, typename Descriptor>
    auto operator()(Policy&& policy, Descriptor&& desc) {
        using ops_t = Ops<std::decay_t<Object>, std::decay_t<Descriptor>>;
        using input_t = typename ops_t::input_t;
        return ops_error_handling_dispatcher{}(std::forward<Policy>(policy), [&]() {
            return ops_t{}(std::forward<Policy>(policy), std::forward<Descriptor>(desc), input_t{});
        });
    }

    template <typename Policy, typename Descriptor, typename Head, typename... Tail>
    auto operator()(Policy&& policy, Descriptor&& desc, Head&& head, Tail&&... tail) {
        using ops_t = Ops<std::decay_t<Object>, std::decay_t<Descriptor>>;
        using dispatcher_t = ops_input_dispatcher<std::decay_t<Head>, ops_t>;
        return dispatcher_t{}(std::forward<Policy>(policy),
                              std::forward<Descriptor>(desc),
                              std::forward<Head>(head),
                              std::forward<Tail>(tail)...);
    }
};

template <typename Object, typename T, template <typename, typename> typename Ops>
struct ops_policy_dispatcher_object<Object, T, Ops, /* IsPolicy = */ false> {
    template <typename Descriptor>
    auto operator()(Descriptor&& desc) {
        using ops_t = Ops<std::decay_t<Object>, std::decay_t<Descriptor>>;
        using input_t = typename ops_t::input_t;
        return ops_t{}(host_policy::get_default(), std::forward<Descriptor>(desc), input_t{});
    }

    template <typename Descriptor, typename Head, typename... Tail>
    auto operator()(Descriptor&& desc, Head&& head, Tail&&... tail) {
        using ops_t = Ops<std::decay_t<Object>, std::decay_t<Descriptor>>;
        using dispatcher_t = ops_input_dispatcher<std::decay_t<Head>, ops_t>;
        return dispatcher_t{}(host_policy::get_default(),
                              std::forward<Descriptor>(desc),
                              std::forward<Head>(head),
                              std::forward<Tail>(tail)...);
    }
};

} // namespace v1

using v1::ops_input_dispatcher;
using v1::ops_policy_dispatcher;
using v1::ops_policy_dispatcher_object;

} // namespace oneapi::dal::detail
