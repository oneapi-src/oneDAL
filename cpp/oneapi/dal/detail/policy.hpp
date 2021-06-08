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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/communicator.hpp"

namespace oneapi::dal::detail {
namespace v1 {

class host_policy_impl;
class data_parallel_policy_impl;
class spmd_policy_impl;

template <typename T>
struct is_execution_policy : std::bool_constant<false> {};

template <typename T>
constexpr bool is_execution_policy_v = is_execution_policy<T>::value;

template <typename T>
struct is_local_policy : std::bool_constant<false> {};

template <typename T>
constexpr bool is_local_policy_v = is_local_policy<T>::value;

template <typename T>
struct is_distributed_policy : std::bool_constant<false> {};

template <typename T>
constexpr bool is_distributed_policy_v = is_distributed_policy<T>::value;

template <typename T>
struct is_host_policy : std::bool_constant<false> {};

template <typename T>
constexpr bool is_host_policy_v = is_host_policy<T>::value;

template <typename T>
struct is_data_parallel_policy : std::bool_constant<false> {};

template <typename T>
constexpr bool is_data_parallel_policy_v = is_data_parallel_policy<T>::value;

enum class cpu_extension : uint64_t {
    none = 0U,
    sse2 = 1U << 0,
    ssse3 = 1U << 1,
    sse42 = 1U << 2,
    avx = 1U << 3,
    avx2 = 1U << 4,
    avx512 = 1U << 5
};

class ONEDAL_EXPORT default_host_policy {};

class ONEDAL_EXPORT host_policy : public base {
public:
    host_policy();

    static const host_policy& get_default() {
        const static host_policy instance;
        return instance;
    }

    cpu_extension get_enabled_cpu_extensions() const noexcept;

    auto& set_enabled_cpu_extensions(const cpu_extension& extensions) {
        set_enabled_cpu_extensions_impl(extensions);
        return *this;
    }

private:
    void set_enabled_cpu_extensions_impl(const cpu_extension& extensions) noexcept;

    pimpl<host_policy_impl> impl_;
};

template <>
struct is_execution_policy<host_policy> : std::bool_constant<true> {};

template <>
struct is_local_policy<host_policy> : std::bool_constant<true> {};

template <>
struct is_host_policy<host_policy> : std::bool_constant<true> {};

#ifdef ONEDAL_DATA_PARALLEL
class ONEDAL_EXPORT data_parallel_policy : public base {
public:
    data_parallel_policy(const sycl::queue& queue) : queue_(queue) {
        init_impl(queue);
    }

    sycl::queue& get_queue() const noexcept {
        return queue_;
    }

    host_policy to_host() const {
        return host_policy::get_default();
    }

private:
    void init_impl(const sycl::queue& queue);

private:
    mutable sycl::queue queue_;
    pimpl<data_parallel_policy_impl> impl_;
};
#endif

#ifdef ONEDAL_DATA_PARALLEL
template <>
struct is_execution_policy<data_parallel_policy> : std::bool_constant<true> {};
#endif

#ifdef ONEDAL_DATA_PARALLEL
template <>
struct is_local_policy<data_parallel_policy> : std::bool_constant<true> {};
#endif

#ifdef ONEDAL_DATA_PARALLEL
template <>
struct is_data_parallel_policy<data_parallel_policy> : std::bool_constant<true> {};
#endif

class ONEDAL_EXPORT spmd_policy_base : public base {
public:
    explicit spmd_policy_base(const spmd_communicator& comm);

    const spmd_communicator& get_communicator() const;

private:
    pimpl<spmd_policy_impl> impl_;
};

template <typename LocalPolicy>
class spmd_policy : public spmd_policy_base {
    static_assert(is_execution_policy_v<LocalPolicy>, "Unknown local policy type");

public:
    explicit spmd_policy(const LocalPolicy& local_policy, const spmd_communicator& comm)
            : spmd_policy_base(comm),
              local_policy_(local_policy) {}

    const LocalPolicy& get_local() const {
        return local_policy_;
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T = LocalPolicy, std::enable_if<is_data_parallel_policy_v<T>>* = nullptr>
    spmd_policy<host_policy> to_host() const {
        return spmd_policy<host_policy>{ local_policy_.to_host(), get_communicator() };
    }
#endif

#ifdef ONEDAL_DATA_PARALLEL
    template <typename T = LocalPolicy, std::enable_if<is_data_parallel_policy_v<T>>* = nullptr>
    sycl::queue get_queue() const {
        return get_local().get_queue();
    }
#endif

private:
    LocalPolicy local_policy_;
};

template <typename LocalPolicy>
struct is_execution_policy<spmd_policy<LocalPolicy>> : std::bool_constant<true> {};

template <typename LocalPolicy>
struct is_distributed_policy<spmd_policy<LocalPolicy>> : std::bool_constant<true> {};

template <typename LocalPolicy>
struct is_host_policy<spmd_policy<LocalPolicy>> : is_host_policy<LocalPolicy> {};

template <typename LocalPolicy>
struct is_data_parallel_policy<spmd_policy<LocalPolicy>> : is_data_parallel_policy<LocalPolicy> {};

using spmd_host_policy = spmd_policy<host_policy>;

#ifdef ONEDAL_DATA_PARALLEL
using spmd_data_parallel_policy = spmd_policy<data_parallel_policy>;
#endif

struct internal_policy_accessor {
    template <typename, typename = void>
    struct can_get_internal : std::false_type {};

    template <typename T>
    struct can_get_internal<T, std::void_t<decltype(std::declval<T>().get_internal_policy())>>
            : std::true_type {};

    template <typename T>
    static constexpr bool can_get_internal_v = can_get_internal<T>::value;

    template <typename T>
    static auto get_internal(T&& object) {
        return object.get_internal_policy();
    }
};

template <typename T>
inline constexpr bool can_get_internal_policy_v =
    internal_policy_accessor::template can_get_internal_v<T>;

template <typename T>
inline constexpr bool can_cast_to_internal_policy_v =
    is_execution_policy_v<T> || can_get_internal_policy_v<T>;

template <typename T>
inline auto get_internal_policy(T&& object) {
    static_assert(can_get_internal_policy_v<T>);
    return internal_policy_accessor::get_internal(object);
}

template <typename T>
using internal_policy_type_t = decltype(get_internal_policy(std::declval<T>()));

template <typename T, std::enable_if_t<is_execution_policy_v<T>>* = nullptr>
inline T cast_to_internal_policy(const T& object) {
    return object;
}

template <typename T, std::enable_if_t<can_get_internal_policy_v<T>>* = nullptr>
inline internal_policy_type_t<T> cast_to_internal_policy(const T& object) {
    return get_internal_policy(object);
}

} // namespace v1

using v1::is_execution_policy_v;
using v1::is_local_policy_v;
using v1::is_distributed_policy_v;
using v1::is_host_policy_v;
using v1::is_data_parallel_policy_v;
using v1::can_get_internal_policy_v;
using v1::can_cast_to_internal_policy_v;

using v1::cpu_extension;
using v1::default_host_policy;
using v1::host_policy;
using v1::spmd_policy;
using v1::spmd_host_policy;
using v1::internal_policy_accessor;
using v1::internal_policy_type_t;

using v1::get_internal_policy;
using v1::cast_to_internal_policy;

#ifdef ONEDAL_DATA_PARALLEL
using v1::data_parallel_policy;
using v1::spmd_data_parallel_policy;
#endif

} // namespace oneapi::dal::detail
