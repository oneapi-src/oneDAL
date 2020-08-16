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

#include "oneapi/dal/io/csv_data_source/detail/read_ops.hpp"
#include "oneapi/dal/io/csv_data_source/read_types.hpp"
#include "oneapi/dal/read.hpp"

namespace oneapi::dal::detail {

template <typename table, typename Descriptor>
struct read_ops<table, Descriptor, dal::csv_data_source::detail::tag> : dal::csv_data_source::detail::read_ops<table, Descriptor> {};

} // namespace oneapi::dal::detail
