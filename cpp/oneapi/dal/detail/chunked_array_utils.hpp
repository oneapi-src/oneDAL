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

#include <variant>

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/memory.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/array_impl.hpp"
#include "oneapi/dal/detail/array_utils.hpp"
#include "oneapi/dal/detail/chunked_array_impl.hpp"

namespace oneapi::dal::detail {
namespace v2 {

template <typename T>
void copy(array_impl<T>& dst_impl, const chunked_array_base& src) {
    auto* const dst_raw = dst_impl.get_mutable_data();
    auto* const dst_byte = reinterpret_cast<dal::byte_t*>(dst_raw);

    const auto chunk_count = src.get_chunk_count();
    const auto full_size = src.get_size_in_bytes();
    const auto dst_policy_var = dst_impl.get_policy();
    ONEDAL_ASSERT(dst_impl.get_size_in_bytes() == full_size);

    if (full_size == std::int64_t{ 0l }) return;

    std::int64_t offset = 0l;
    for (std::int64_t c = 0l; c < chunk_count; ++c) {
        const auto& chunk = src.get_chunk_impl(c);

        const auto chunk_size = chunk.get_size_in_bytes();
        const auto src_policy_var = chunk.get_policy();

        const auto* const src_ptr = chunk.get_data();
        auto* const dst_ptr = dst_byte + offset;

        const auto* const src_ptr_raw = reinterpret_cast<const void*>(src_ptr);
        auto* const dst_ptr_raw = reinterpret_cast<void*>(dst_ptr);

        ONEDAL_ASSERT(src_ptr_raw != nullptr);

        const auto copy_chunk = [&](const auto& dst_policy, const auto& src_policy) {
            memcpy(dst_policy, src_policy, dst_ptr_raw, src_ptr_raw, chunk_size);
        };

        std::visit(copy_chunk, dst_policy_var, src_policy_var);

        offset += chunk_size;
    }

    ONEDAL_ASSERT(offset == full_size);
}

template <typename T>
void copy(dal::array<T>& dst, const chunked_array_base& src) {
    constexpr detail::pimpl_accessor accessor;
    auto dst_impl = accessor.get_pimpl(dst);
    return copy(*dst_impl, src);
}

} // namespace v2

using v2::copy;

} // namespace oneapi::dal::detail
