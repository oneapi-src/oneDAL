/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

enum graph_status {
    ok = 0, /*!< No error found*/
    bad_arguments = -5, /*!< Bad argument(s) passed*/
    bad_allocation = -11, /*!< Memory allocation error*/
};

enum register_size { r8 = 1, r16 = 2, r32 = 4, r64 = 8, r128 = 16, r256 = 32, r512 = 64 };

// 1/64 for memory capacity and ~0.005 for cpu.
constexpr double graph_storage_divider_by_density = 0.015625;

} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
