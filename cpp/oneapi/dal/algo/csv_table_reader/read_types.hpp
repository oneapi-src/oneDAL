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

namespace oneapi::dal::csv_table_reader {

namespace detail {
template<typename Object>
class read_input_impl;
template<typename Object>
class read_result_impl;

template<>
class read_input_impl<table>;
template<>
class read_result_impl<table>;
} // namespace detail

template<typename Object>
class read_input;

template<>
class ONEAPI_DAL_EXPORT read_input<table> : public base {
public:
    read_input(const char * file_name);

    const char * get_file_name() const;

    auto& set_file_name(const char * file_name) {
        set_file_name_impl(file_name);
        return *this;
    }

private:
    void set_file_name_impl(const char * file_name);

    dal::detail::pimpl<detail::read_input_impl<table>> impl_;
};

template<typename Object>
class read_result;

template<>
class ONEAPI_DAL_EXPORT read_result<table> {
public:
    read_result();

    table get_table() const;

    auto& set_table(const table& value) {
        set_table_impl(value);
        return *this;
    }

private:
    void set_table_impl(const table&);

    dal::detail::pimpl<detail::read_result_impl<table>> impl_;
};

} // namespace oneapi::dal::pca
