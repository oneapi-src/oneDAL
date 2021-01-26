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

#include "oneapi/dal/detail/memory_impl_host.hpp"

#include <daal/include/services/daal_memory.h>

namespace oneapi::dal::detail::v1 {

template <typename AllocOp>
void* alloc_impl(AllocOp&& op, std::size_t size, std::size_t alignment) {
    auto ptr = op(size, alignment);
    if (ptr == nullptr) {
        throw dal::host_bad_alloc();
    }
    return ptr;
}

void* malloc(const default_host_policy&, std::size_t size) {
    return alloc_impl(daal::services::daal_malloc, size, daal::DAAL_MALLOC_DEFAULT_ALIGNMENT);
}

void* calloc(const default_host_policy&, std::size_t size) {
    return alloc_impl(daal::services::daal_calloc, size, daal::DAAL_MALLOC_DEFAULT_ALIGNMENT);
}

void free(const default_host_policy&, void* pointer) {
    daal::services::daal_free(pointer);
}

void fill(const default_host_policy&,
          void* dest,
          std::size_t dest_size,
          const void* pattern,
          std::size_t pattern_size) {
    // TODO: can be optimized in future

    ONEDAL_ASSERT(dest_size % pattern_size == 0);

    auto dest_bytes = static_cast<std::uint8_t*>(dest);
    auto pattern_bytes = static_cast<const std::uint8_t*>(pattern);

    for (std::size_t i = 0; i < dest_size; i++) {
        dest_bytes[i] = pattern_bytes[i % pattern_size];
    }
}

void memset(const default_host_policy&, void* dest, std::int32_t value, std::int64_t size) {
    // TODO: can be optimized in future
    std::memset(dest, value, detail::integral_cast<std::size_t>(size));
}

void memcpy(const default_host_policy&, void* dest, const void* src, std::int64_t size) {
    const size_t converted_size = detail::integral_cast<std::size_t>(size);
    std::int32_t status =
        daal::services::internal::daal_memcpy_s(dest, converted_size, src, converted_size);
    if (status) {
        throw dal::internal_error(detail::error_messages::unknown_memcpy_error());
    }
}

} // namespace oneapi::dal::detail::v1
