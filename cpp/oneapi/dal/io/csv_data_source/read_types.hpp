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

#include "oneapi/dal/io/csv_data_source/common.hpp"

namespace oneapi::dal::csv_data_source {

namespace detail {
template<typename Object>
class read_input_impl;

template<>
class read_input_impl<table>;
} // namespace detail

template<typename Object>
class read_input;

template<>
class ONEAPI_DAL_EXPORT read_input<table> : public base {
public:
    read_input();

private:
    dal::detail::pimpl<detail::read_input_impl<table>> impl_;
};

} // namespace oneapi::dal::pca
