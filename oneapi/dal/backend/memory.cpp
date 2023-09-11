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

#include <daal/include/services/daal_memory.h>
#include "oneapi/dal/backend/memory.hpp"

namespace oneapi::dal::backend {

void memcpy(void* dest, const void* src, std::int64_t size) {
    ONEDAL_ASSERT(dest != nullptr);
    ONEDAL_ASSERT(src != nullptr);

    const std::size_t converted_size = detail::integral_cast<std::size_t>(size);
    std::int32_t status =
        daal::services::internal::daal_memcpy_s(dest, converted_size, src, converted_size);
    if (status) {
        throw dal::internal_error(detail::error_messages::unknown_memcpy_error());
    }
}

} // namespace oneapi::dal::backend
