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

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/backend/common.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"

namespace oneapi::dal::backend::interop {

template <typename Op, typename OnUnknown>
constexpr auto inline dispatch_by_table_type(Op&& op, OnUnknown&& on_unknown, data_type dtype) {
    switch (dtype) {
        case data_type::int32: return op(std::int32_t{});
        case data_type::float64: return op(double{});
        case data_type::float32: return op(float{});
        default: return on_unknown(dtype);
    }
}

} // namespace oneapi::dal::backend::interop
