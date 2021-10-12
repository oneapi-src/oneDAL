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

#include <cstring>

#include "oneapi/dal/detail/memory_impl_dpc.hpp"
#include "oneapi/dal/detail/memory_impl_host.hpp"
#include "oneapi/dal/common.hpp"

namespace oneapi::dal::detail {
namespace v1 {

template <typename T>
class empty_delete {
public:
    void operator()(T*) const noexcept {}
};

template <typename T, typename Policy>
class default_delete {
public:
    explicit default_delete(const Policy& policy) : policy_(policy) {}

    void operator()(T* data) const {
        detail::free(policy_, data);
    }

private:
    std::remove_reference_t<Policy> policy_;
};

template <typename T>
inline auto make_default_delete(const detail::default_host_policy& policy) {
    return default_delete<T, detail::default_host_policy>{ policy };
}

#ifdef ONEDAL_DATA_PARALLEL

template <typename T>
inline auto make_default_delete(const detail::data_parallel_policy& policy) {
    return default_delete<T, detail::data_parallel_policy>{ policy };
}

#endif

} // namespace v1

using v1::empty_delete;
using v1::default_delete;
using v1::make_default_delete;

} // namespace oneapi::dal::detail

namespace oneapi::dal::preview::detail {

using namespace std;

template <typename T, typename Allocator>
class destroy_delete {
public:
    explicit destroy_delete(std::int64_t count, Allocator& alloc) : count_(count), alloc_(alloc) {}

    template <typename T_ = T, std::enable_if_t<!is_trivial<T_>::value, bool> = true>
    void operator()(T* data) {
        for (std::int64_t i = 0; i < count_; ++i) {
            data[i].~T();
        }
        oneapi::dal::preview::detail::deallocate(alloc_, data, count_);
    }

    template <typename T_ = T, std::enable_if_t<is_trivial<T_>::value, bool> = true>
    void operator()(T* data) {
        oneapi::dal::preview::detail::deallocate(alloc_, data, count_);
    }

private:
    std::int64_t count_;
    Allocator alloc_;
};

struct byte_alloc_iface {
    virtual ~byte_alloc_iface() = default;
    virtual byte_t* allocate(std::int64_t n) = 0;
    virtual void deallocate(byte_t* ptr, std::int64_t n) = 0;
};

template <typename Alloc>
struct alloc_connector : public byte_alloc_iface {
    using allocator_traits_t =
        typename std::allocator_traits<Alloc>::template rebind_traits<byte_t>;
    using t_byte_allocator = typename std::allocator_traits<Alloc>::template rebind_alloc<byte_t>;
    alloc_connector(Alloc alloc) : alloc_(alloc) {}
    byte_t* allocate(std::int64_t count) override {
        auto ptr = allocator_traits_t::allocate(alloc_, count);
        if (ptr == nullptr) {
            throw host_bad_alloc();
        }
        return ptr;
    };

    void deallocate(byte_t* ptr, std::int64_t count) override {
        if (ptr != nullptr) {
            allocator_traits_t::deallocate(alloc_, ptr, count);
        }
    };

private:
    t_byte_allocator alloc_;
};

} // namespace oneapi::dal::preview::detail
