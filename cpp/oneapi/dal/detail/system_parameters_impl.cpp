/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include "oneapi/dal/detail/cpu_info_impl.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/system_parameters_impl.hpp"
#include <daal/src/services/service_defines.h>
#include <daal/include/services/internal/daal_kernel_defines.h>

#include <sstream>

namespace oneapi::dal::detail {
namespace v1 {

system_parameters_impl::system_parameters_impl() {
    using daal::services::Environment;
    Environment* env = Environment::getInstance();
    sys_info_["top_enabled_cpu_extension"] =
        from_daal_cpu_type(DAAL_KERNEL_BUILD_MAX_INSTRUCTION_SET_ID);
    sys_info_["max_number_of_threads"] = static_cast<std::uint32_t>(env->getNumberOfThreads());
}

cpu_extension system_parameters_impl::get_top_enabled_cpu_extension() const {
    return std::any_cast<cpu_extension>(sys_info_.find("top_enabled_cpu_extension")->second);
}

std::uint32_t system_parameters_impl::get_max_number_of_threads() const {
    return std::any_cast<std::uint32_t>(sys_info_.find("max_number_of_threads")->second);
}

void system_parameters_impl::print_any(const std::any& value, std::ostringstream& ss) const {
    const std::type_info& ti = value.type();
    if (ti == typeid(cpu_extension)) {
        ss << to_string(std::any_cast<cpu_extension>(value));
    }
    else if (ti == typeid(std::uint32_t)) {
        ss << std::any_cast<std::uint32_t>(value);
    }
    else {
        throw unimplemented{ dal::detail::error_messages::unsupported_data_type() };
    }
}

std::string system_parameters_impl::dump() const {
    std::ostringstream ss;
    for (auto it = sys_info_.begin(); it != sys_info_.end(); ++it) {
        ss << it->first << " : ";
        print_any(it->second, ss);
        ss << "; ";
    }
    return std::move(ss).str();
}

} // namespace v1
} // namespace oneapi::dal::detail
