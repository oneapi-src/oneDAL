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

#include <utility>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/common.hpp"

#include <daal/include/services/daal_shared_ptr.h>
#include <daal/include/services/internal/buffer.h>

namespace oneapi::dal::backend::interop {

template <typename DataType>
using buffer_t = daal::services::internal::Buffer<DataType>;

template <typename DataType>
ONEDAL_EXPORT buffer_t<DataType> convert(const dal::array<DataType>& array);

template <typename DataType>
ONEDAL_EXPORT auto convert_with_status(const dal::array<DataType>& array)
    -> std::pair<buffer_t<DataType>, daal::services::Status>;

} // namespace oneapi::dal::backend::interop
