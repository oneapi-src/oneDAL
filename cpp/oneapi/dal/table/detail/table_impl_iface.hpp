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

#include "oneapi/dal/table/detail/access_iface.hpp"

namespace oneapi::dal {
namespace v1 {
class table_metadata;
enum class data_layout;
} // namespace v1

using v1::table_metadata;
using v1::data_layout;

} // namespace oneapi::dal

namespace oneapi::dal::detail {

class table_impl_iface : public access_provider_iface {
public:
    virtual std::int64_t get_column_count() const = 0;
    virtual std::int64_t get_row_count() const = 0;
    virtual const table_metadata& get_metadata() const = 0;
    virtual std::int64_t get_kind() const = 0;
    virtual data_layout get_data_layout() const = 0;
};

class homogen_table_impl_iface : public table_impl_iface {
public:
    virtual const void* get_data() const = 0;
};

} // namespace oneapi::dal::detail
