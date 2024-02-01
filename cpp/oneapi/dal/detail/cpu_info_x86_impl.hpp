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

#include <daal/src/services/service_defines.h>

#include <any>
#include <map>
#include <string>
#include <sstream>

#include <iostream>

namespace oneapi::dal::detail {
namespace v1 {

class cpu_info_x86 : public cpu_info_iface {
public:
    cpu_info_x86() {
        info_["cpu_extensions"] = backend::detect_top_cpu_extension();
        info_["vendor"] = (daal_check_is_intel_cpu() ? cpu_vendor::intel : cpu_vendor::amd);
    }

    cpu_info_x86(const detail::cpu_extension cpu_extension) {
        info_["cpu_extensions"] = cpu_extension;
        info_["vendor"] = (daal_check_is_intel_cpu() ? cpu_vendor::intel : cpu_vendor::amd);
    }

    cpu_vendor get_cpu_vendor() const override {
        return std::any_cast<detail::cpu_vendor>(info_.find("vendor")->second);
    }

    detail::cpu_extension get_cpu_extensions() const override {
        return std::any_cast<detail::cpu_extension>(info_.find("cpu_extensions")->second);
    }

    std::string dump() const override {
        std::stringstream ss;
        for (auto it = info_.begin(); it != info_.end(); ++it) {
            ss << it->first << ":";
            print_any(it->second, ss);
            ss << "; ";
        }
        std::string result;
        std::string token;
        while (ss >> token) {
            result += token + " ";
        }
        return result;
    }

private:
    std::map<std::string, std::any> info_;

    template <typename T>
    void print(const std::any& value, std::stringstream& ss) const {
        T typed_value = std::any_cast<T>(typed_value);
        ss << to_string(typed_value);
    }

    void print_any(const std::any& value, std::stringstream& ss) const {
        const std::type_info& ti = value.type();
        if (ti == typeid(detail::cpu_extension)) {
            print<detail::cpu_extension>(value, ss);
        }
        else if (ti == typeid(detail::cpu_vendor)) {
            print<detail::cpu_vendor>(value, ss);
        }
    }
};

} // namespace v1
using v1::cpu_info_iface;
} // namespace oneapi::dal::detail
