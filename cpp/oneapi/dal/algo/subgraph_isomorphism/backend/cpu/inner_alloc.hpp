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

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/memory.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

using byte_alloc_iface_t = oneapi::dal::preview::detail::byte_alloc_iface;

struct inner_alloc {
    inner_alloc(byte_alloc_iface_t* byte_allocator) : byte_allocator_(byte_allocator) {}
    inner_alloc(const byte_alloc_iface_t* byte_allocator)
            : byte_allocator_(const_cast<byte_alloc_iface_t*>(byte_allocator)) {}

    template <typename T>
    T* allocate(std::int64_t n) {
        T* const ptr = reinterpret_cast<T*>(byte_allocator_->allocate(n * sizeof(T)));
        if (!ptr) {
            throw oneapi::dal::host_bad_alloc();
        }
        return ptr;
    }

    template <typename T>
    void deallocate(T* ptr, std::int64_t n) {
        byte_allocator_->deallocate(reinterpret_cast<byte_t*>(ptr), n * sizeof(T));
        return;
    }

    template <typename T>
    oneapi::dal::detail::shared<T> make_shared_memory(std::int64_t n) {
        const auto ptr = oneapi::dal::detail::shared<T>(allocate<T>(n), [=](T* p) {
            deallocate<T>(p, n);
        });
        return ptr;
    }

    byte_alloc_iface_t* get_byte_allocator() {
        return byte_allocator_;
    }

    const byte_alloc_iface_t* get_byte_allocator() const {
        return byte_allocator_;
    }

private:
    byte_alloc_iface_t* byte_allocator_;
};

} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
