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

#pragma once

#include "oneapi/dal/detail/cpu.hpp"

#include <string>

namespace oneapi::dal::detail {
namespace v1 {

class cpu_info_iface {
public:
    /// The CPU vendor
    virtual cpu_vendor get_cpu_vendor() const = 0;

    /// The highest supported CPU extension
    virtual cpu_extension get_top_cpu_extension() const = 0;

    /// The dump of all supported CPU features in the format:
    /// feature 1: value1; feature2: value2; ...
    virtual std::string dump() const = 0;

    virtual ~cpu_info_iface() {}
};

} // namespace v1
using v1::cpu_info_iface;
} // namespace oneapi::dal::detail
