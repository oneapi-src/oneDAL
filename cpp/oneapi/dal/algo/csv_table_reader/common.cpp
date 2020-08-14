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

#include "oneapi/dal/algo/csv_table_reader/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::csv_table_reader {

class detail::descriptor_impl : public base {
public:
    char delimiter    = ',';
    bool parse_header = false;
};

using detail::descriptor_impl;

descriptor_base::descriptor_base() : impl_(new descriptor_impl{}) {}

char descriptor_base::get_delimiter() const {
    return impl_->delimiter;
}

bool descriptor_base::get_parse_header() const {
    return impl_->parse_header;
}

void descriptor_base::set_delimiter_impl(char value) {
    impl_->delimiter = value;
}

void descriptor_base::set_parse_header_impl(bool value) {
    impl_->parse_header = value;
}

} // namespace oneapi::dal::csv_table_reader
