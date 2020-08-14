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

#include "oneapi/dal/algo/csv_table_reader/read_types.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::csv_table_reader {

class detail::read_input_impl : public base {
public:
    read_input_impl(const input_stream& stream) : stream(stream) {}

    input_stream stream;
};

class detail::read_result_impl : public base {
public:
    table value;
};

using detail::read_input_impl;
using detail::read_result_impl;

read_input::read_input(const input_stream& stream) : impl_(new read_input_impl(stream)) {}

input_stream read_input::get_input_stream() const {
    return impl_->stream;
}

void read_input::set_input_stream_impl(const input_stream& stream) {
    impl_->stream = stream;
}

read_result::read_result() : impl_(new read_result_impl{}) {}

table read_result::get_table() const {
    return impl_->value;
}

void read_result::set_table_impl(const table& value) {
    impl_->value = value;
}

} // namespace oneapi::dal::pca
