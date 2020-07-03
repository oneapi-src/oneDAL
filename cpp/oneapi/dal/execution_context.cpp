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

#include "oneapi/dal/execution_context.hpp"
#include "oneapi/dal/backend/interop/common.hpp"

namespace oneapi::dal {

class detail::default_execution_context_impl : public base {
public:
    cpu_extension cpu_extensions_mask = backend::interop::detect_top_cpu_extension();
};

using detail::default_execution_context_impl;

default_execution_context::default_execution_context()
        : impl_(new default_execution_context_impl()) {}

void default_execution_context::set_enabled_cpu_extensions_impl(
    const cpu_extension& extensions) noexcept {
    impl_->cpu_extensions_mask = extensions;
}

cpu_extension default_execution_context::get_enabled_cpu_extensions() const noexcept {
    return impl_->cpu_extensions_mask;
}

} // namespace oneapi::dal
