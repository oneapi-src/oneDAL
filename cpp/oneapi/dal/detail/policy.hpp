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
#ifdef ONEAPI_DAL_DATA_PARALLEL
#include <CL/sycl.hpp>
#endif

#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::detail {

class host_policy_impl;
class data_parallel_policy_impl;

enum class cpu_extension : uint64_t {
    none = 0U,
    sse2 = 1U << 0,
    ssse3 = 1U << 1,
    sse42 = 1U << 2,
    avx = 1U << 3,
    avx2 = 1U << 4,
    avx512 = 1U << 5
};

class ONEAPI_DAL_EXPORT default_host_policy {};

class ONEAPI_DAL_EXPORT host_policy : public base {
public:
    host_policy();

    cpu_extension get_enabled_cpu_extensions() const noexcept;

    auto& set_enabled_cpu_extensions(const cpu_extension& extensions) {
        set_enabled_cpu_extensions_impl(extensions);
        return *this;
    }

private:
    void set_enabled_cpu_extensions_impl(const cpu_extension& extensions) noexcept;

    pimpl<detail::host_policy_impl> impl_;
};

#ifdef ONEAPI_DAL_DATA_PARALLEL
// to detail
class ONEAPI_DAL_EXPORT data_parallel_policy : public base {
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

template <typename T>
struct is_execution_policy : std::bool_constant<false> {};

template <>
struct is_execution_policy<host_policy> : std::bool_constant<true> {};

#ifdef ONEAPI_DAL_DATA_PARALLEL
template <>
struct is_execution_policy<data_parallel_policy> : std::bool_constant<true> {};
#endif

template <typename T>
constexpr bool is_execution_policy_v = is_execution_policy<T>::value;

} // namespace oneapi::dal::detail
