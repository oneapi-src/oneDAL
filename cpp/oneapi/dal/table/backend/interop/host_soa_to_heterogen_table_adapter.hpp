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

#pragma once

#include <daal/include/data_management/data/soa_numeric_table.h>

#include "oneapi/dal/table/heterogen.hpp"
#include "oneapi/dal/table/detail/metadata_utils.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/daal_object_owner.hpp"
#include "oneapi/dal/table/backend/interop/host_soa_table_adapter.hpp"

namespace oneapi::dal::backend::interop {

ONEDAL_EXPORT heterogen_table convert_to_heterogen(const soa_table_t& t);
ONEDAL_EXPORT heterogen_table convert_to_heterogen(const soa_table_ptr_t& t);

} // namespace oneapi::dal::backend::interop
