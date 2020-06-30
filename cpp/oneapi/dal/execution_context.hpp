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

#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal {

/* Forward declarations */
class default_execution_context;
class data_parallel_execution_context;

namespace detail {
struct execution_context_tag {};
class default_execution_context_impl;
} // namespace detail

enum class cpu_extension : uint64_t {
    none   = 0U,
    sse2   = 1U << 0,
    ssse3  = 1U << 1,
    sse42  = 1U << 2,
    avx    = 1U << 3,
    avx2   = 1U << 4,
    avx512 = 1U << 5
};

class ONEAPI_DAL_EXPORT default_execution_context : public base {
public:
    using tag_t = detail::execution_context_tag;
    default_execution_context();

    cpu_extension get_enabled_cpu_extensions() const noexcept;

    auto& set_enabled_cpu_extensions(const cpu_extension& extensions) noexcept {
        set_enabled_cpu_extensions_impl(extensions);
        return *this;
    }

private:
    void set_enabled_cpu_extensions_impl(const cpu_extension& extensions) noexcept;
    dal::detail::pimpl<detail::default_execution_context_impl> impl_;
};

inline auto make_context() {
    return default_execution_context();
}

} // namespace oneapi::dal
