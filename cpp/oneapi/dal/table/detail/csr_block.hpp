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

#include "oneapi/dal/array.hpp"

namespace oneapi::dal::detail {
namespace v1 {

/// Type of csr indexing, currently it's implemented only `one_based` indexing.
enum class csr_indexing { zero_based, one_based };

/// @tparam T The type of data values in block.
///           CSR block supports at least :expr:`float`, :expr:`double`, and :expr:`std::int32_t` types of :literal:`T`.
template <typename T>
struct csr_block {
    dal::array<T> data;
    dal::array<std::int64_t> column_indices;
    dal::array<std::int64_t> row_indices;

    csr_block() : data(), column_indices(), row_indices() {}
};
} // namespace v1

using v1::csr_block;
using v1::csr_indexing;

} // namespace oneapi::dal::detail
