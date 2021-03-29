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

#include "oneapi/dal/array.hpp"

namespace oneapi::dal::detail {
namespace v1 {

template <typename T>
class array_via_policy {
public:
    array_via_policy() = delete;

    template <typename... Args>
    static array<T> wrap(const default_host_policy& policy, Args&&... args) {
        return array<T>{ std::forward<Args>(args)... };
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename... Args>
    static array<T> wrap(const data_parallel_policy& policy, Args&&... args) {
        return array<T>{ policy.get_queue(), std::forward<Args>(args)... };
    }
#endif
};

template <typename T, typename Body>
inline auto dispath_by_policy(const array<T>& data, Body&& body) {
#ifdef ONEDAL_DATA_PARALLEL
    const auto optional_queue = data.get_queue();
    if (optional_queue) {
        return body(data_parallel_policy{ optional_queue.value() });
    }
    else {
        return body(default_host_policy{});
    }
#else
    return body(default_host_policy{});
#endif
}

template <typename T, typename Y>
struct array_reinterpret_cast_op {
    const T* operator()(const Y* ptr) const {
        return reinterpret_cast<const T*>(ptr);
    }

    T* operator()(Y* ptr) const {
        return reinterpret_cast<T*>(ptr);
    }
};

template <typename T, typename U, typename CastOp>
inline array<T> array_cast_generic(const array<U>& ary, const CastOp& cast_op) {
    if (ary.get_size() % sizeof(T) > 0) {
        throw invalid_argument{ error_messages::incompatible_array_reinterpret_cast_types() };
    }

    const std::int64_t new_count = ary.get_size() / sizeof(T);
    if (ary.has_mutable_data()) {
        T* mutable_ptr = cast_op(ary.get_mutable_data());
        return array<T>{ ary, mutable_ptr, new_count };
    }
    else {
        const T* immutable_ptr = cast_op(ary.get_data());
        return array<T>{ ary, immutable_ptr, new_count };
    }
}

template <typename T, typename U>
inline array<T> reinterpret_array_cast(const array<U>& ary) {
    return array_cast_generic<T>(ary, array_reinterpret_cast_op<T, U>{});
}

template <typename T>
inline array<T> discard_mutable_data(const array<T>& ary) {
    if (!ary.has_mutable_data()) {
        return ary;
    }
    return array<T>{ ary, ary.get_data(), ary.get_count() };
}

} // namespace v1

using v1::array_via_policy;
using v1::dispath_by_policy;
using v1::reinterpret_array_cast;
using v1::discard_mutable_data;

} // namespace oneapi::dal::detail
