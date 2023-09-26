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

#include "oneapi/dal/table/backend/empty_table_impl.hpp"
#include "oneapi/dal/table/backend/homogen_table_impl.hpp"
#include "oneapi/dal/table/backend/heterogen_table_impl.hpp"
#include "oneapi/dal/table/backend/csr_table_impl.hpp"

using oneapi::dal::backend::empty_table_impl;
using oneapi::dal::backend::homogen_table_impl;
using oneapi::dal::backend::csr_table_impl;

ONEDAL_REGISTER_SERIALIZABLE(empty_table_impl)
ONEDAL_REGISTER_SERIALIZABLE(homogen_table_impl)
ONEDAL_REGISTER_SERIALIZABLE(csr_table_impl)
ONEDAL_REGISTER_SERIALIZABLE_INIT(tables)

// TODO: Implement serialization for heterogen table
using oneapi::dal::backend::heterogen_table_impl;
ONEDAL_REGISTER_SERIALIZABLE(heterogen_table_impl)
