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

#include "oneapi/dal/algo/csv_table_reader/common.hpp"
#include "oneapi/dal/data/input_stream.hpp"

namespace oneapi::dal::csv_table_reader {

namespace detail {
class read_input_impl;
class read_result_impl;
} // namespace detail

class ONEAPI_DAL_EXPORT read_input : public base {
public:
    read_input(const input_stream& stream);

    input_stream get_input_stream() const;

    auto& set_input_stream(const input_stream& stream) {
        set_input_stream_impl(stream);
        return *this;
    }

private:
    void set_input_stream_impl(const input_stream& stream);

    dal::detail::pimpl<detail::read_input_impl> impl_;
};

class ONEAPI_DAL_EXPORT read_result {
public:
    read_result();

    table get_table() const;

    auto& set_table(const table& value) {
        set_table_impl(value);
        return *this;
    }

private:
    void set_table_impl(const table&);

    dal::detail::pimpl<detail::read_result_impl> impl_;
};

} // namespace oneapi::dal::pca
