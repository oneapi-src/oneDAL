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

#include <variant>

#include "oneapi/dal/detail/memory.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::detail {
namespace v1 {

template <typename T>
class array_impl : public base {
    using cshared = detail::shared<const T>;
    using shared = detail::shared<T>;

    template <typename U>
    friend class array_impl;

public:
    template <typename Policy, typename Allocator>
    static array_impl<T>* empty(const Policy& policy, std::int64_t count, const Allocator& alloc) {
        auto data = alloc.allocate(count);
        return new array_impl<T>{ data, count, [alloc, count](T* ptr) {
                                     alloc.deallocate(ptr, count);
                                 } };
    }

    template <typename Policy, typename K, typename Allocator>
    static array_impl<T>* full(const Policy& policy,
                               std::int64_t count,
                               K&& element,
                               const Allocator& alloc) {
        auto array = empty(policy, count, alloc);
        detail::fill(policy, array->get_mutable_data(), count, std::forward<K>(element));
        return array;
    }

public:
    array_impl() : count_(0) {}

    template <typename Deleter>
    array_impl(T* data, std::int64_t count, Deleter&& d) {
        reset(data, count, std::forward<Deleter>(d));
    }

    template <typename ConstDeleter>
    array_impl(const T* data, std::int64_t count, ConstDeleter&& d) {
        reset(data, count, std::forward<ConstDeleter>(d));
    }

    template <typename Y, typename K>
    array_impl(const array_impl<Y>& ref, K* data, std::int64_t count) {
        reset(ref, data, count);
    }

    std::int64_t get_count() const noexcept {
        return count_;
    }

    const T* get_data() const noexcept {
        if (const auto& mut_ptr = std::get_if<shared>(&data_owned_)) {
            return mut_ptr->get();
        }
        else {
            const auto& immut_ptr = std::get<cshared>(data_owned_);
            return immut_ptr.get();
        }
    }

    T* get_mutable_data() const {
        try {
            const auto& mut_ptr = std::get<shared>(data_owned_);
            return mut_ptr.get();
        }
        catch (std::bad_variant_access&) {
            throw internal_error(
                dal::detail::error_messages::array_does_not_contain_mutable_data());
        }
    }

    bool has_mutable_data() const noexcept {
        return std::holds_alternative<shared>(data_owned_) && (get_mutable_data() != nullptr);
    }

    template <typename Policy, typename Allocator>
    void need_mutable_data(const Policy& policy, const Allocator& alloc) {
        if (has_mutable_data() || count_ == 0) {
            return;
        }
        else {
            auto immutable_data = get_data();
            auto copy_data = alloc.allocate(count_);
            detail::memcpy(policy, copy_data, immutable_data, sizeof(T) * count_);

            reset(copy_data, count_, [alloc, count = this->count_](T* ptr) {
                alloc.deallocate(ptr, count);
            });
            return;
        }
    }

    void reset() {
        data_owned_ = std::variant<cshared, shared>();
        count_ = 0;
    }

    template <typename Policy, typename Allocator>
    void reset(const Policy& policy, std::int64_t count, const Allocator& alloc) {
        auto new_data = alloc.allocate(count);
        reset(new_data, count, [alloc, count](T* ptr) {
            alloc.deallocate(ptr, count);
        });
    }

    template <typename Deleter>
    void reset(T* data, std::int64_t count, Deleter&& deleter) {
        data_owned_ = shared(data, std::forward<Deleter>(deleter));
        count_ = count;
    }

    template <typename ConstDeleter>
    void reset(const T* data, std::int64_t count, ConstDeleter&& deleter) {
        data_owned_ = cshared(data, std::forward<ConstDeleter>(deleter));
        count_ = count;
    }

    template <typename Y>
    void reset(const array_impl<Y>& ref, T* data, std::int64_t count) {
        if (ref.has_mutable_data()) {
            data_owned_ = shared(std::get<1>(ref.data_owned_), data);
        }
        else {
            data_owned_ = shared(std::get<0>(ref.data_owned_), data);
        }
        count_ = count;
    }

    template <typename Y>
    void reset(const array_impl<Y>& ref, const T* data, std::int64_t count) {
        if (ref.has_mutable_data()) {
            data_owned_ = cshared(std::get<1>(ref.data_owned_), data);
        }
        else {
            data_owned_ = cshared(std::get<0>(ref.data_owned_), data);
        }
        count_ = count;
    }

private:
    std::variant<cshared, shared> data_owned_;
    std::int64_t count_;
};

} // namespace v1

using v1::array_impl;

} // namespace oneapi::dal::detail
