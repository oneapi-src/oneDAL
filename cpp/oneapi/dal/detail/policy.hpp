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

#include <type_traits>
#ifdef ONEDAL_DATA_PARALLEL
#include <sycl/sycl.hpp>
#endif

#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::detail {
namespace v1 {

class host_policy_impl;
class data_parallel_policy_impl;

template <typename T>
struct is_execution_policy : std::bool_constant<false> {};

template <typename T>
struct is_local_policy : std::bool_constant<false> {};

template <typename T>
struct is_distributed_policy : std::bool_constant<false> {};

template <typename T>
struct is_host_policy : std::bool_constant<false> {};

template <typename T>
struct is_data_parallel_policy : std::bool_constant<false> {};

template <typename T>
inline constexpr bool is_execution_policy_v = is_execution_policy<T>::value;

template <typename T>
inline constexpr bool is_local_policy_v = is_local_policy<T>::value;

template <typename T>
inline constexpr bool is_distributed_policy_v = is_distributed_policy<T>::value;

template <typename T>
inline constexpr bool is_host_policy_v = is_host_policy<T>::value;

template <typename T>
inline constexpr bool is_data_parallel_policy_v = is_data_parallel_policy<T>::value;

enum class cpu_extension : uint64_t {
    none = 0U,
    sse2 = 1U << 0,
    sse42 = 1U << 2,
    avx2 = 1U << 4,
    avx512 = 1U << 5
};

struct ONEDAL_EXPORT threading_policy {
    bool thread_pinning;
    int max_threads_per_core;

    threading_policy(bool thread_pinning_ = false, int max_threads_per_core_ = 0)
            : thread_pinning(thread_pinning_),
              max_threads_per_core(max_threads_per_core_) {}
};

class ONEDAL_EXPORT host_policy : public base {
public:
    host_policy(bool thread_pinning = false, int max_threads_per_core = 0);

    static const host_policy& get_default() {
        const static host_policy instance;
        return instance;
    }

    cpu_extension get_enabled_cpu_extensions() const noexcept;
    bool get_thread_pinning() const noexcept;
    int get_max_threads_per_core() const noexcept;
    threading_policy get_threading_policy() const noexcept;

    auto& set_enabled_cpu_extensions(const cpu_extension& extensions) {
        set_enabled_cpu_extensions_impl(extensions);
        return *this;
    }

    auto& set_thread_pinning(const bool& thread_pinning) {
        set_thread_pinning_impl(thread_pinning);
        return *this;
    }

    auto& set_max_threads_per_core(const int& max_threads_per_core) {
        set_max_threads_per_core_impl(max_threads_per_core);
        return *this;
    }

private:
    void set_enabled_cpu_extensions_impl(const cpu_extension& extensions) noexcept;
    void set_thread_pinning_impl(const bool& thread_pinning) noexcept;
    void set_max_threads_per_core_impl(const int& max_threads_per_core) noexcept;

    pimpl<host_policy_impl> impl_;
};

class ONEDAL_EXPORT default_host_policy : public host_policy {
public:
    default_host_policy() : host_policy{ host_policy::get_default() } {}
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

} // namespace v1

using v1::is_execution_policy_v;
using v1::is_local_policy_v;
using v1::is_distributed_policy_v;
using v1::is_host_policy_v;
using v1::is_data_parallel_policy_v;

using v1::cpu_extension;
using v1::default_host_policy;
using v1::host_policy;
using v1::threading_policy;

#ifdef ONEDAL_DATA_PARALLEL
using v1::data_parallel_policy;
#endif

} // namespace oneapi::dal::detail
