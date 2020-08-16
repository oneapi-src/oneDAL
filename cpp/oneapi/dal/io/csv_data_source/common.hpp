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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::csv_data_source {

namespace detail {
struct tag {};
class params_impl;
} // namespace detail

class ONEAPI_DAL_EXPORT params_base : public base {
public:
    using tag_t    = detail::tag;

    params_base(const char * file_name);

    auto get_delimiter() const -> char;
    auto get_parse_header() const -> bool;
    auto get_file_name() const -> const char *;

protected:
    void set_delimiter_impl(char value);
    void set_parse_header_impl(bool value);
    void set_file_name_impl(const char *);

    dal::detail::pimpl<detail::params_impl> impl_;
};

class params : public params_base {
public:

    params(const char * file_name) : params_base(file_name) {}

    auto& set_delimiter(char value) {
        set_delimiter_impl(value);
        return *this;
    }

    auto& set_parse_header(bool value) {
        set_parse_header_impl(value);
        return *this;
    }

    auto& set_file_name(const char * value) {
        set_file_name_impl(value);
        return *this;
    }
};

} // namespace oneapi::dal::csv_data_source
