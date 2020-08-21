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

#include "oneapi/dal/io/csv/read_types.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/memory.hpp"
#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::csv {

template <>
class detail::read_args_impl<table> : public base {
public:
    read_args_impl() {}
};

read_args<table>::read_args() : impl_(new detail::read_args_impl<table>()) {}

} // namespace oneapi::dal::csv
