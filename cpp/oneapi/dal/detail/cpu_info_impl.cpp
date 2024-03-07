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
    std::string vendor_str;
    switch (vendor) {
        case cpu_vendor::unknown: vendor_str = std::string("Unknown"); break;
        case cpu_vendor::intel: vendor_str = std::string("Intel"); break;
        case cpu_vendor::amd: vendor_str = std::string("AMD"); break;
        case cpu_vendor::arm: vendor_str = std::string("ARM"); break;
    }
    return vendor_str;
}

std::string to_string(cpu_extension extension) {
    std::string extension_str;
    switch (extension) {
        case cpu_extension::none: extension_str = std::string("none"); break;
#if defined(TARGET_X86_64)
        case cpu_extension::sse2: extension_str = std::string("sse2"); break;
        case cpu_extension::sse42: extension_str = std::string("sse42"); break;
        case cpu_extension::avx2: extension_str = std::string("avx2"); break;
        case cpu_extension::avx512: extension_str = std::string("avx512"); break;
#elif defined(TARGET_ARM)
        case cpu_extension::sve: extension_str = std::string("sve"); break;
#endif
    }
    return extension_str;
}

} // namespace v1
using v1::to_string;
} // namespace oneapi::dal::detail
