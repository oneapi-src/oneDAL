/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#include "oneapi/dal/algo/subgraph_isomorphism/detail/byte_alloc.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {
struct byte_alloc_iface;
}

#if defined(__INTEL_COMPILER)
#define ONEAPI_RESTRICT
//restrict
#else
#define ONEAPI_RESTRICT
#endif

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

enum register_size { r8 = 1, r16 = 2, r32 = 4, r64 = 8, r128 = 16, r256 = 32, r512 = 64 };

const std::int64_t null_node = 0xffffffffffffffff; /*!< Null node value*/

// 1/64 for memory capacity and ~0.005 for cpu.
const double graph_storage_divider_by_density = 0.015625;

enum graph_storage_scheme { auto_detect, bit, list };

struct inner_alloc {
    using byte_t = char;

    inner_alloc(detail::byte_alloc_iface* byte_allocator) : byte_allocator_(byte_allocator) {}
    inner_alloc(const detail::byte_alloc_iface* byte_allocator)
            : byte_allocator_(const_cast<detail::byte_alloc_iface*>(byte_allocator)) {}

    template <typename T>
    T* allocate(std::int64_t n) {
        return reinterpret_cast<T*>(byte_allocator_->allocate(n * sizeof(T)));
    }

    template <typename T>
    void deallocate(T* ptr, std::int64_t n) {
        return byte_allocator_->deallocate(reinterpret_cast<byte_t*>(ptr), n * sizeof(T));
    }

    template <typename T>
    oneapi::dal::detail::shared<T> make_shared_memory(std::int64_t n) {
        return oneapi::dal::detail::shared<T>(allocate<T>(n), [=](T* p) {
            deallocate<T>(p, n);
        });
    }

    detail::byte_alloc_iface* get_byte_allocator() {
        return byte_allocator_;
    }

    const detail::byte_alloc_iface* get_byte_allocator() const {
        return byte_allocator_;
    }

private:
    detail::byte_alloc_iface* byte_allocator_;
};

} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
