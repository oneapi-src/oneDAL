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

#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

struct byte_alloc_iface {
    using byte_t = char;
    virtual byte_t* allocate(std::int64_t n) = 0;
    virtual void deallocate(byte_t* ptr, std::int64_t n) = 0;
};

template <typename Alloc>
struct alloc_connector : public byte_alloc_iface {
    using byte_t = char;
    using t_allocator_traits =
        typename std::allocator_traits<Alloc>::template rebind_traits<byte_t>;
    alloc_connector(Alloc alloc) : _alloc(alloc) {}
    byte_t* allocate(std::int64_t count) override {
        typename t_allocator_traits::pointer ptr = t_allocator_traits::allocate(_alloc, count);
        if (ptr == nullptr) {
            throw host_bad_alloc();
        }
        return ptr;
    };

    void deallocate(byte_t* ptr, std::int64_t count) override {
        if (ptr != nullptr) {
            t_allocator_traits::deallocate(_alloc, ptr, count);
        }
    };

private:
    Alloc _alloc;
};

} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
