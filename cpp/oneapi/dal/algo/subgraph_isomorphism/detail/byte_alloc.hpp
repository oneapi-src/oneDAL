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
    virtual byte_t* allocate(std::int64_t n) = 0;
    virtual void deallocate(byte_t* ptr, std::int64_t n) = 0;
};

template <typename Alloc>
struct alloc_connector : public byte_alloc_iface {
    using allocator_traits_t =
        typename std::allocator_traits<Alloc>::template rebind_traits<byte_t>;
    alloc_connector(Alloc alloc) : alloc_(alloc) {}
    byte_t* allocate(std::int64_t count) override {
        typename allocator_traits_t::pointer ptr = allocator_traits_t::allocate(alloc_, count);
        return ptr;
    };

    void deallocate(byte_t* ptr, std::int64_t count) override {
        if (ptr != nullptr) {
            allocator_traits_t::deallocate(alloc_, ptr, count);
        }
    };

private:
    Alloc alloc_;
};

} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
