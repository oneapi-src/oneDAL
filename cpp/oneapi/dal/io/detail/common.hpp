/* file: common.hpp */
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
#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::preview::detail {

template <typename T, typename Allocator>
class vector {
public:
    using data_t = T;
    using impl_t = array<data_t>;
    using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<T>;
    using allocator_traits = typename std::allocator_traits<Allocator>::template rebind_traits<T>;
    using empty_delete = dal::detail::empty_delete<const T>;

    vector() : impl_(new impl_t()), allocator(allocator_type()) {
        impl_->reset(allocator_traits::allocate(allocator, 1), 0, empty_delete{});
    }

    vector(Allocator& a) : vector() {
        allocator = a;
    }

    vector(std::int64_t count) : vector() {
        capacity = count;
        impl_->reset(allocator_traits::allocate(allocator, capacity), count, empty_delete{});
    }

    vector(std::int64_t count, Allocator& a) : vector(a) {
        capacity = count;
        impl_->reset(allocator_traits::allocate(allocator, capacity), count, empty_delete{});
    }

    virtual ~vector() {
        if (impl_->get_mutable_data() != nullptr) {
            allocator_traits::deallocate(allocator, impl_->get_mutable_data(), capacity);
        }
    }

    const T& operator[](std::int64_t index) const noexcept {
        return impl_->get_data()[index];
    }

    T& operator[](std::int64_t index) noexcept {
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

    const T* get_data() const noexcept {
        return impl_->get_data();
    }

    T* get_mutable_data() const {
        return impl_->get_mutable_data();
    }

    vector(vector<T, Allocator>&& other)
            : impl_(other.impl_),
              capacity(other.capacity),
              allocator(other.allocator) {}

    vector<T, Allocator> operator=(vector<T, Allocator>&& other) {
        swap(*this, other);
        return *this;
    }

    vector(const vector<T, Allocator>& other)
            : impl_(other.impl_),
              capacity(other.capacity),
              allocator(other.allocator) {}

    vector<T, Allocator> operator=(const vector<T, Allocator>& other) {
        vector<T, Allocator> tmp{ other };
        swap(*this, tmp);
        return *this;
    }

    constexpr void resize(std::int64_t count) {
        if (count <= capacity) {
            impl_->reset(impl_->get_mutable_data(), count, empty_delete{});
        }
        else {
            auto count_temp = count;
            auto old_capacity = capacity;
            while (count_temp != 0) {
                count_temp = count_temp >> 1;
                capacity *= 2;
            }
            auto data_ptr = allocator_traits::allocate(allocator, capacity);
            for (std::int64_t i = 0; i < this->get_count(); i++) {
                data_ptr[i] = operator[](i);
            }
            allocator_traits::deallocate(allocator, impl_->get_mutable_data(), old_capacity);
            impl_->reset(data_ptr, count, empty_delete{});
        }
    }

    constexpr void reserve(std::int64_t new_capacity) {
        if (new_capacity > capacity) {
            auto data_ptr = allocator_traits::allocate(allocator, new_capacity);
            for (std::int64_t i = 0; i < this->get_count(); i++) {
                data_ptr[i] = operator[](i);
            }
            allocator_traits::deallocate(allocator, impl_->get_mutable_data(), capacity);
            impl_->reset(data_ptr, impl_->get_count(), empty_delete{});
            capacity = new_capacity;
        }
    }

    constexpr void push_back(const T& value) {
        if (impl_->get_count() + 1 <= capacity) {
            impl_->reset(impl_->get_mutable_data(), impl_->get_count() + 1, empty_delete{});
        }
        else {
            resize(impl_->get_count() + 1);
        }
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

template <typename T = std::int32_t, typename Allocator = std::allocator<char>>
using edge_list_container = vector<T, Allocator>;

} // namespace oneapi::dal::preview::detail
