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

#pragma once

#include "oneapi/dal/array.hpp"

#if defined(__INTEL_COMPILER)
#define PRAGMA_IVDEP         _Pragma("ivdep")
#define PRAGMA_VECTOR_ALWAYS _Pragma("vector always")
#else
#define PRAGMA_IVDEP
#define PRAGMA_VECTOR_ALWAYS
#endif

namespace oneapi::dal::preview::detail {

template <typename T>
using container = dal::array<T>;

template <typename T, typename Allocator = std::allocator<char>>
class vector_container {
public:
    using data_t = T;
    using impl_t = array<data_t>;
    using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<T>;
    using allocator_traits = typename std::allocator_traits<Allocator>::template rebind_traits<T>;
    using empty_delete = dal::detail::empty_delete<const T>;

    vector_container() : impl_(new impl_t()), allocator(allocator_type()) {
        impl_->reset(allocator_traits::allocate(allocator, capacity), 0, empty_delete{});
    }

    vector_container(std::int64_t count) : vector_container() {
        //TODO: add check for count
        capacity = count;
        impl_->reset(allocator_traits::allocate(allocator, capacity), count, empty_delete{});
    }

    vector_container(const allocator_type& a) : impl_(new impl_t()), allocator(a) {
        allocator = a;
        impl_->reset(allocator_traits::allocate(allocator, capacity), 0, empty_delete{});
    }

    vector_container(std::int64_t count, const allocator_type& a) : vector_container(a) {
        //TODO: add check for count
        resize(count);
    }

    virtual ~vector_container() {
        if (impl_->get_mutable_data() != nullptr) {
            allocator_traits::deallocate(allocator, impl_->get_mutable_data(), capacity);
        }
    }

    const T& operator[](std::int64_t index) const noexcept {
        //TODO: add check for index
        return impl_->get_data()[index];
    }

    T& operator[](std::int64_t index) noexcept {
        //TODO: add check for index
        return impl_->get_mutable_data()[index];
    }

    std::int64_t get_count() const noexcept {
        return impl_->get_count();
    }

    std::int64_t size() const noexcept {
        return impl_->get_count();
    }

    std::int64_t get_size() const noexcept {
        return impl_->get_size();
    }

    const allocator_type& get_allocator() const noexcept {
        return allocator;
    }

    const T* get_data() const noexcept {
        return impl_->get_data();
    }

    T* get_mutable_data() const {
        return impl_->get_mutable_data();
    }

    vector_container(vector_container<T, Allocator>&& other)
            : impl_(other.impl_),
              capacity(other.capacity),
              allocator(other.allocator) {}

    vector_container<T, Allocator> operator=(vector_container<T, Allocator>&& other) {
        swap(*this, other);
        return *this;
    }

    vector_container(const vector_container<T, Allocator>& other)
            : impl_(other.impl_),
              capacity(other.capacity),
              allocator(other.allocator) {}

    vector_container<T, Allocator> operator=(const vector_container<T, Allocator>& other) {
        vector_container<T, Allocator> tmp{ other };
        swap(*this, tmp);
        return *this;
    }

    constexpr void resize(std::int64_t count) {
        if (count > capacity) {
            std::int64_t count_temp = count;
            std::int64_t new_capacity = capacity;
            while (count_temp != 0) { // new_capacity = nearest old_capacity * 2^k > count
                count_temp = count_temp >> 1;
                new_capacity *= 2;
            }
            reserve(new_capacity);
            impl_->reset(impl_->get_mutable_data(), count, empty_delete{});
        }
        impl_->reset(impl_->get_mutable_data(), count, empty_delete{});
    }

    constexpr void reserve(std::int64_t new_capacity) {
        if (new_capacity > capacity) {
            T* data_ptr = allocator_traits::allocate(allocator, new_capacity);
            T* old_data_ptr = impl_->get_mutable_data();
            const std::int64_t old_count = get_count();
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for (std::int64_t i = 0; i < old_count; i++) {
                data_ptr[i] = old_data_ptr[i];
            }

            allocator_traits::deallocate(allocator, old_data_ptr, capacity);
            impl_->reset(data_ptr, impl_->get_count(), empty_delete{});
            capacity = new_capacity;
        }
    }

    constexpr void push_back(const T& value) {
        resize(impl_->get_count() + 1);
        operator[](impl_->get_count() - 1) = value;
    }

    using iterator = T*;
    iterator begin() {
        return impl_->get_mutable_data();
    }
    iterator end() {
        return impl_->get_mutable_data() + impl_->get_count();
    }

private:
    using pimpl = dal::detail::pimpl<impl_t>;

    pimpl impl_;

    allocator_type allocator;

    std::int64_t capacity = 1;

    friend dal::detail::pimpl_accessor;
};

} // namespace oneapi::dal::preview::detail
