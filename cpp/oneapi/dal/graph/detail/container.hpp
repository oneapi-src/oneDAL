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

#include <tuple>
#include <new>

#include "oneapi/dal/array.hpp"

namespace oneapi::dal::preview::detail {

template <typename T, typename Allocator = std::allocator<char>>
class vector_container;

template <typename T>
inline void copy(const T* old_begin, const T* old_end, T* new_begin) {
    const std::int64_t count = std::distance(old_begin, old_end);
    for (std::int64_t i = 0; i < count; i++) {
        new_begin[i] = old_begin[i];
    }
}

template <typename T>
inline void fill(T* begin, T* end, const T& value) {
    const std::int64_t count = std::distance(begin, end);
    for (std::int64_t i = 0; i < count; i++) {
        begin[i] = value;
    }
}

template <typename T>
inline void copy(const T& from, T& to) {
    to = from;
}

template <typename First, typename Second, typename Third>
inline void copy(const std::tuple<First, Second, Third>* old_begin,
                 const std::tuple<First, Second, Third>* old_end,
                 std::tuple<First, Second, Third>* new_begin) {
    const std::int64_t count = std::distance(old_begin, old_end);
    for (std::int64_t i = 0; i < count; i++) {
        const auto& b = old_begin[i];
        auto& a = new_begin[i];
        std::get<0>(a) = std::get<0>(b);
        std::get<1>(a) = std::get<1>(b);
        std::get<2>(a) = std::get<2>(b);
    }
}

template <typename First, typename Second, typename Third>
inline void copy(const std::tuple<First, Second, Third>& from,
                 std::tuple<First, Second, Third>& to) {
    std::get<0>(to) = std::get<0>(from);
    std::get<1>(to) = std::get<1>(from);
    std::get<2>(to) = std::get<2>(from);
}

template <typename T>
using container = dal::array<T>;

template <typename T, typename Alloc>
struct construct {
    template <typename T_ = T, std::enable_if_t<!is_trivial<T_>::value, bool> = true>
    void operator()(T* data_ptr, std::int64_t capacity, const Alloc& a) {
        for (std::int64_t i = 0; i < capacity; ++i) {
            new (data_ptr + i) T();
        }
    }

    template <typename T_ = T, std::enable_if_t<is_trivial<T_>::value, bool> = true>
    void operator()(T* data_ptr, std::int64_t capacity, const Alloc& a) {}
};

template <typename T, typename InnerAlloc, typename OuterAlloc>
struct construct<vector_container<T, InnerAlloc>, OuterAlloc> {
    void operator()(vector_container<T, InnerAlloc>* data_ptr,
                    std::int64_t capacity,
                    const OuterAlloc& a) {
        using data_t = vector_container<T, InnerAlloc>;
        InnerAlloc inner_a(a);
        for (std::int64_t i = 0; i < capacity; ++i) {
            new (data_ptr + i) data_t(inner_a);
        }
    }
};

template <typename T, typename Allocator>
class vector_container {
public:
    using data_t = T;
    using impl_t = array<data_t>;
    using allocator_type = typename std::allocator_traits<Allocator>::template rebind_alloc<T>;

    vector_container() : impl_(new impl_t()), allocator_(allocator_type()) {
        T* data_ptr = oneapi::dal::preview::detail::allocate(allocator_, capacity_);
        construct<data_t, allocator_type>{}(data_ptr, capacity_, allocator_);
        impl_->reset(data_ptr,
                     capacity_,
                     destroy_delete<data_t, allocator_type>(capacity_, allocator_));
    }

    vector_container(std::int64_t count, const allocator_type& a) : vector_container(a) {
        this->resize(count);
    }

    vector_container(std::int64_t count) : vector_container(count, allocator_type()) {
        this->resize(count);
    }

    vector_container(const allocator_type& a) : impl_(new impl_t()), allocator_(a) {
        T* data_ptr = oneapi::dal::preview::detail::allocate(allocator_, capacity_);
        construct<data_t, allocator_type>{}(data_ptr, capacity_, allocator_);
        impl_->reset(data_ptr,
                     capacity_,
                     destroy_delete<data_t, allocator_type>(capacity_, allocator_));
    }

    vector_container(std::int64_t count, const T& value, const allocator_type& a)
            : vector_container(count, a) {
        fill(impl_->get_mutable_data(), impl_->get_mutable_data() + capacity_, value);
    }

    vector_container(const vector_container<T, Allocator>& other)
            : impl_(new impl_t()),
              allocator_(other.get_allocator()) {
        capacity_ = other.capacity();
        count_ = other.size();
        T* data_ptr = oneapi::dal::preview::detail::allocate(allocator_, capacity_);
        construct<data_t, allocator_type>{}(data_ptr, capacity_, allocator_);
        const T* other_data_ptr = other.get_data();
        preview::detail::copy(other_data_ptr, other_data_ptr + count_, data_ptr);
        impl_->reset(data_ptr,
                     capacity_,
                     destroy_delete<data_t, allocator_type>(capacity_, allocator_));
    }

    vector_container<T, Allocator> operator=(const vector_container<T, Allocator>& other) {
        pimpl tmp_impl(new impl_t());
        std::swap(this->impl_, tmp_impl);
        this->allocator_ = other.get_allocator();
        capacity_ = other.capacity();
        count_ = other.size();
        T* data_ptr = oneapi::dal::preview::detail::allocate(allocator_, capacity_);
        construct<data_t, allocator_type>{}(data_ptr, capacity_, allocator_);
        const T* other_data_ptr = other.get_data();
        preview::detail::copy(other_data_ptr, other_data_ptr + count_, data_ptr);
        impl_->reset(data_ptr,
                     capacity_,
                     destroy_delete<data_t, allocator_type>(capacity_, allocator_));

        return *this;
    }

    virtual ~vector_container() {}

    const T& operator[](std::int64_t index) const {
        //TODO: add check for index
        return impl_->get_data()[index];
    }

    T& operator[](std::int64_t index) {
        //TODO: add check for index
        return impl_->get_mutable_data()[index];
    }

    std::int64_t get_count() const noexcept {
        return count_;
    }

    std::int64_t size() const noexcept {
        return count_;
    }

    std::int64_t capacity() const noexcept {
        return capacity_;
    }

    bool empty() const noexcept {
        return (count_ == 0);
    }

    const allocator_type& get_allocator() const noexcept {
        return allocator_;
    }

    const T* get_data() const noexcept {
        return impl_->get_data();
    }

    T* get_mutable_data() const {
        return impl_->get_mutable_data();
    }

    constexpr void resize(std::int64_t count) {
        if (count > capacity_) {
            std::int64_t count_temp = count;
            std::int64_t new_capacity = 1;
            while (count_temp != 0) {
                count_temp = count_temp >> 1;
                new_capacity *= 2;
            }
            reserve(new_capacity);
        }
        this->count_ = count;
    }

    constexpr void reserve(std::int64_t new_capacity) {
        if (new_capacity > capacity_) {
            T* data_ptr = oneapi::dal::preview::detail::allocate(allocator_, new_capacity);
            construct<data_t, allocator_type>{}(data_ptr, new_capacity, allocator_);
            T* old_data_ptr = impl_->get_mutable_data();
            preview::detail::copy(old_data_ptr, old_data_ptr + count_, data_ptr);
            impl_->reset(data_ptr,
                         new_capacity,
                         destroy_delete<data_t, allocator_type>(new_capacity, allocator_));
            capacity_ = new_capacity;
        }
    }

    constexpr void push_back(const T& value) {
        resize(count_ + 1);
        preview::detail::copy(value, operator[](count_ - 1));
    }

    using iterator = T*;

    iterator begin() {
        return impl_->get_mutable_data();
    }

    iterator end() {
        return impl_->get_mutable_data() + count_;
    }

private:
    using pimpl = dal::detail::pimpl<impl_t>;

    pimpl impl_;

    allocator_type allocator_;

    std::int64_t capacity_ = 1;

    std::int64_t count_ = 0;

    friend dal::detail::pimpl_accessor;
};

} // namespace oneapi::dal::preview::detail
