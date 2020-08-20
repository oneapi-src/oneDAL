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

#include <string>

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::csv {

namespace detail {
struct data_source_tag {};
} // namespace detail

class ONEAPI_DAL_EXPORT data_source : public base {
public:
    using tag_t = detail::data_source_tag;

    data_source(const char *file_name) : file_name_(std::string(file_name)) {}

    data_source(std::string file_name) : file_name_(file_name) {}

    char get_delimiter() const {
        return delimiter_;
    }

    bool get_parse_header() const {
        return parse_header_;
    }

    std::string get_file_name() const {
        return file_name_;
    }

    auto &set_delimiter(char value) {
        delimiter_ = value;
        return *this;
    }

    auto &set_parse_header(bool value) {
        parse_header_ = value;
        return *this;
    }

    auto &set_file_name(const char *value) {
        file_name_ = std::string(value);
        return *this;
    }

    auto &set_file_name(const std::string &value) {
        file_name_ = value;
        return *this;
    }

private:
    char delimiter_        = ',';
    bool parse_header_     = false;
    std::string file_name_ = "";
};

} // namespace oneapi::dal::csv
