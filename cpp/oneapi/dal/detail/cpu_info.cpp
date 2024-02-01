/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "oneapi/dal/detail/cpu_info.hpp"
#include "oneapi/dal/detail/cpu_info_x86_impl.hpp"

#include <daal/src/services/service_defines.h>

namespace oneapi::dal::detail {
namespace v1 {

cpu_info::cpu_info() {
    impl_ = detail::pimpl<cpu_info_iface>(new cpu_info_x86());
}
cpu_info::cpu_info(const detail::cpu_extension cpu_extension_) {
    impl_ = detail::pimpl<cpu_info_iface>(new cpu_info_x86(cpu_extension_));
}

detail::cpu_vendor cpu_info::get_cpu_vendor() const {
    return impl_->get_cpu_vendor();
}

detail::cpu_extension cpu_info::get_cpu_extensions() const {
    return impl_->get_cpu_extensions();
}

std::string cpu_info::dump() const {
    return impl_->dump();
}

} // namespace v1
} // namespace oneapi::dal::detail
