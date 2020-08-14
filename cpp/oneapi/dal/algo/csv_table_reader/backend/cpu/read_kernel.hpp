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

#include "oneapi/dal/algo/csv_table_reader/read_types.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::csv_table_reader::backend {

struct read_kernel_cpu {
    read_result operator()(const dal::backend::context_cpu& ctx,
                           const descriptor_base& params,
                           const read_input& input) const;
};

} // namespace oneapi::dal::csv_table_reader::backend
