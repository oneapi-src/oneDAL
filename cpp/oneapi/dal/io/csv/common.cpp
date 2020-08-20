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

#include "oneapi/dal/io/csv/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::csv {

class detail::data_source_impl : public base {
public:
    char delimiter    = ',';
    bool parse_header = false;
    char *file_name   = nullptr;
};

using detail::data_source_impl;

data_source::data_source(const char *file_name) : impl_(new data_source_impl{}) {
    set_file_name_impl(file_name);
}

data_source::data_source(const std::string &file_name) : impl_(new data_source_impl{}) {
    set_file_name_impl(file_name.c_str());
}

char data_source::get_delimiter() const {
    return impl_->delimiter;
}

bool data_source::get_parse_header() const {
    return impl_->parse_header;
}

const char *data_source::get_file_name() const {
    return impl_->file_name;
}

void data_source::set_delimiter_impl(char value) {
    impl_->delimiter = value;
}

void data_source::set_parse_header_impl(bool value) {
    impl_->parse_header = value;
}

void data_source::set_file_name_impl(const char *value) {
    const size_t len = strlen(value);
    impl_->file_name = new char[len + 1];
    dal::detail::memcpy(dal::detail::default_host_policy{},
                        impl_->file_name,
                        value,
                        sizeof(char) * len);
    impl_->file_name[len] = '\0';
}

} // namespace oneapi::dal::csv
