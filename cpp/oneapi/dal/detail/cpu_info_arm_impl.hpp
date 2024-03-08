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

#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/detail/cpu_info_iface.hpp"

#include <daal/src/services/service_defines.h>

#include <any>
#include <map>
#include <string>
#include <sstream>

namespace oneapi::dal::detail {
namespace v1 {

class cpu_info_arm : public cpu_info_iface {
public:
    cpu_info_arm() {
        info_["top_cpu_extension"] = cpu_extension::sve;
        info_["vendor"] = cpu_vendor::arm);
    }

    cpu_info_arm(const cpu_extension cpu_extension) {
        info_["top_cpu_extension"] = cpu_extension;
        info_["vendor"] = cpu_vendor::arm);
    }

    cpu_vendor get_cpu_vendor() const override {
        return std::any_cast<detail::cpu_vendor>(info_.find("vendor")->second);
    }

    cpu_extension get_top_cpu_extension() const override {
        return std::any_cast<cpu_extension>(info_.find("top_cpu_extension")->second);
    }

    std::string dump() const override {
        std::stringstream ss;
        for (auto it = info_.begin(); it != info_.end(); ++it) {
            ss << it->first << " : ";
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
        T typed_value = std::any_cast<T>(value);
        ss << to_string(typed_value);
    }

    void print_any(const std::any& value, std::stringstream& ss) const {
        const std::type_info& ti = value.type();
        if (ti == typeid(cpu_extension)) {
            print<cpu_extension>(value, ss);
        }
        else if (ti == typeid(cpu_vendor)) {
            print<cpu_vendor>(value, ss);
        }
    }
};

} // namespace v1
using v1::cpu_info_iface;
} // namespace oneapi::dal::detail
