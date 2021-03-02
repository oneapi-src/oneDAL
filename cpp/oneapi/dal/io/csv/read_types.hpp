/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/io/csv/common.hpp"

namespace oneapi::dal::csv {

namespace detail {
namespace v1 {
template <typename Object>
class read_args_impl;
} // namespace v1

using v1::read_args_impl;

} // namespace detail

namespace v1 {

template <typename Object = table>
class read_args;

template <>
class ONEDAL_EXPORT read_args<table> : public base {
public:
    read_args();
    read_args(oneapi::dal::preview::read_mode mode);
    auto& set_read_mode(oneapi::dal::preview::read_mode mode) {
        set_read_mode_impl(mode);
        return *this;
    }
    oneapi::dal::preview::read_mode get_read_mode();

protected:
    void set_read_mode_impl(oneapi::dal::preview::read_mode mode);

private:
    dal::detail::pimpl<detail::read_args_impl<table>> impl_;
};

template <>
class ONEDAL_EXPORT read_args<preview::graph_base> : public base {
public:
    read_args();
    read_args(oneapi::dal::preview::read_mode mode);

private:
    dal::detail::pimpl<detail::read_args_impl<preview::graph_base>> impl_;
};
} // namespace v1

using v1::read_args;

} // namespace oneapi::dal::csv
