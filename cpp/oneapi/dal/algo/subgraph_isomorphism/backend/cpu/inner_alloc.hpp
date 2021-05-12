#pragma once

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/detail/byte_alloc.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

struct byte_alloc_iface;

struct inner_alloc {
    using byte_t = char;

    inner_alloc(byte_alloc_iface* byte_allocator) : byte_allocator_(byte_allocator) {}
    inner_alloc(const byte_alloc_iface* byte_allocator)
            : byte_allocator_(const_cast<byte_alloc_iface*>(byte_allocator)) {}

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

    byte_alloc_iface* get_byte_allocator() {
        return byte_allocator_;
    }

    const byte_alloc_iface* get_byte_allocator() const {
        return byte_allocator_;
    }

private:
    byte_alloc_iface* byte_allocator_;
};

} // namespace oneapi::dal::preview::subgraph_isomorphism::detail