/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::detail {
namespace v1 {

ONEDAL_EXPORT std::int64_t get_empty_table_kind();
ONEDAL_EXPORT std::int64_t get_homogen_table_kind();
ONEDAL_EXPORT std::int64_t get_csr_table_kind();
ONEDAL_EXPORT std::int64_t get_heterogen_table_kind();

} // namespace v1

using v1::get_empty_table_kind;
using v1::get_homogen_table_kind;
using v1::get_csr_table_kind;
using v1::get_heterogen_table_kind;

} // namespace oneapi::dal::detail
