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

#pragma once

#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/detail/cpu_info_iface.hpp"

namespace oneapi::dal::detail {
namespace v1 {

class cpu_info_x86 : public cpu_info_iface {
public:
    cpu_info_x86() {
        cpu_extensions_ = backend::detect_top_cpu_extension();
    }

    cpu_info_x86(const detail::cpu_extension cpu_extension) : cpu_extensions_(cpu_extension) {}

    cpu_vendor get_cpu_vendor() const override {
        return cpu_vendor::intel;
    }

    detail::cpu_extension get_cpu_extensions() const override {
        return cpu_extensions_;
    }

private:
    detail::cpu_extension cpu_extensions_;
};

} // namespace v1
using v1::cpu_info_iface;
} // namespace oneapi::dal::detail
