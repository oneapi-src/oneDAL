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

#include "oneapi/dal/table/detail/table_kinds.hpp"

namespace oneapi::dal::detail {

std::int64_t get_empty_table_kind() {
    return 0l;
}

std::int64_t get_homogen_table_kind() {
    return 1l;
}

std::int64_t get_csr_table_kind() {
    return 10l;
}

std::int64_t get_heterogen_table_kind() {
    return 100l;
}

} // namespace oneapi::dal::detail
