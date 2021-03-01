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
    using empty_delete = dal::detail::empty_delete<const T>;

    vector_container() : impl_(new impl_t()), allocator_(allocator_type()) {
        T* data_ptr = oneapi::dal::preview::detail::allocate(allocator_, capacity_);
        impl_->reset(data_ptr, capacity_, empty_delete{});
    }

    vector_container(const allocator_type& a) : impl_(new impl_t()), allocator_(a) {
        T* data_ptr = oneapi::dal::preview::detail::allocate(allocator_, capacity_);
        impl_->reset(data_ptr, capacity_, empty_delete{});
    }

    virtual ~vector_container() {
        if (impl_->has_mutable_data()) {
            oneapi::dal::preview::detail::deallocate(allocator_,
                                                     impl_->get_mutable_data(),
                                                     capacity_);
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
        return count_;
    }

    std::int64_t size() const noexcept {
        return count_;
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
            T* old_data_ptr = impl_->get_mutable_data();
            const std::int64_t old_count = count_;
            PRAGMA_IVDEP
            PRAGMA_VECTOR_ALWAYS
            for (std::int64_t i = 0; i < old_count; i++) {
                data_ptr[i] = old_data_ptr[i];
            }

            impl_->reset(data_ptr, new_capacity, empty_delete{});
            oneapi::dal::preview::detail::deallocate(allocator_, old_data_ptr, capacity_);
            capacity_ = new_capacity;
        }
    }

    constexpr void push_back(const T& value) {
        resize(count_ + 1);
        operator[](count_ - 1) = value;
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
