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

#include "oneapi/dal/detail/cpu_info_iface.hpp"

namespace oneapi::dal::detail {
namespace v1 {

std::string to_string(cpu_vendor vendor) {
    switch (vendor) {
        case cpu_vendor::unknown: return std::string("unknown");
        case cpu_vendor::intel: return std::string("intel");
        case cpu_vendor::amd: return std::string("amd");
        case cpu_vendor::arm: return std::string("arm");
        default: break; /// error handling
    }
    return std::string();
}

std::string to_string(cpu_extension extension) {
    switch (extension) {
        case cpu_extension::none: return std::string("none");
        case cpu_extension::sse2: return std::string("sse2");
        case cpu_extension::sse42: return std::string("sse42");
        case cpu_extension::avx2: return std::string("avx2");
        case cpu_extension::avx512: return std::string("avx512");
        default: break; /// error handling
    }
    return std::string();
}

} // namespace v1
using v1::to_string;
} // namespace oneapi::dal::detail
