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

#include <algorithm>
#include <stdexcept> // TODO: change by onedal exceptions
#include <variant>

#include "oneapi/dal/memory.hpp"

namespace oneapi::dal {

template <typename T>
class array {
    static_assert(!std::is_const_v<T>, "array class cannot have const-qualified type of data");

    template <typename U>
    friend class array;

public:
    using data_t = T;

public:
    static array<T> empty(std::int64_t count) {
        return empty(detail::cpu_dispatch_default{}, count);
    }

    template <typename Policy, typename AllocKind=default_parameter_tag>
    static array<T> empty(const Policy& policy,
                          std::int64_t count,
                          const AllocKind& alloc = {}) {
        auto data = detail::malloc<T>(policy, count, alloc);
        return array<T>{ data, count, make_default_delete<T>(policy) };
    }

    template <typename K>
    static array<T> full(std::int64_t count, K&& element) {
        return full(detail::cpu_dispatch_default{}, count, std::forward<K>(element));
    }

    template <typename Policy, typename K, typename AllocKind=default_parameter_tag>
    static array<T> full(const Policy& policy,
                         std::int64_t count,
                         K&& element,
                         const AllocKind& alloc = {}) {
        auto array = empty(policy, count, alloc);
        detail::fill(policy, array.get_mutable_data(), count, std::forward<K>(element));
        return array;
    }

    static array<T> zeros(std::int64_t count) {
        // TODO: can be optimized in future
        return full(detail::cpu_dispatch_default{}, count, T{});
    }

    template <typename Policy, typename AllocKind=default_parameter_tag>
    static array<T> zeros(const Policy& policy,
                          std::int64_t count,
                          const AllocKind& alloc = {}) {
        // TODO: can be optimized in future
        return full(policy, count, T{}, alloc);
    }

    static array<T> wrap(const T* data, std::int64_t count) {
        return { data, count, empty_delete<const T>{} };
    }

    static array<T> wrap(T* data, std::int64_t count) {
        return { data, count, empty_delete<T>{} };
    }

public:
    array() : data_owned_ptr_(nullptr), data_owned_const_ptr_(nullptr), count_(0) {}

    template <typename Deleter>
    array(T* data, std::int64_t count, Deleter&& deleter) {
        reset(data, count, std::forward<Deleter>(deleter));
    }

    template <typename Deleter>
    array(const T* data, std::int64_t count, Deleter&& deleter) {
        reset(data, count, std::forward<Deleter>(deleter));
    }

    template <typename Y, typename K>
    array(const array<Y>& ref, K* data, std::int64_t count)
            : data_owned_ptr_(ref.data_owned_ptr_, nullptr),
              data_owned_const_ptr_(nullptr),
              data_(data),
              count_(count) {}

    T* get_mutable_data() const {
        return std::get<T*>(data_); // TODO: convert to dal exception
    }

    const T* get_data() const {
        if (auto ptr_val = std::get_if<T*>(&data_)) {
            return *ptr_val;
        }
        else {
            return std::get<const T*>(data_);
        }
    }

    bool has_mutable_data() const {
        return std::holds_alternative<T*>(data_) && (get_mutable_data() != nullptr);
    }

    array& need_mutable_data() {
        return need_mutable_data(detail::cpu_dispatch_default{});
    }

    template <typename Policy, typename AllocKind = default_parameter_tag>
    array& need_mutable_data(const Policy& policy, const AllocKind& alloc = {}) {
        if (has_mutable_data() || count_ == 0) {
            return *this;
        } else {
            auto immutable_data = get_data();
            auto copy_data      = detail::malloc<T>(policy, count_, alloc);
            detail::memcpy(policy, copy_data, immutable_data, sizeof(T) * count_);

            reset(copy_data, count_, make_default_delete<T>(policy));
            return *this;
        }
    }

    std::int64_t get_count() const {
        return count_;
    }

    std::int64_t get_size() const {
        return count_ * sizeof(T);
    }

    void reset() {
        data_owned_ptr_.reset();
        data_owned_const_ptr_.reset();
        data_  = std::variant<T*, const T*>();
        count_ = 0;
    }

    void reset(std::int64_t count) {
        reset(detail::cpu_dispatch_default{}, count);
    }

    template <typename Policy, typename AllocKind=default_parameter_tag>
    void reset(const Policy& policy, std::int64_t count, const AllocKind& alloc = {}) {
        auto new_data = detail::malloc<T>(policy, count, alloc);
        reset(new_data, count, make_default_delete<T>(policy));
    }

    template <typename Deleter>
    void reset(T* data, std::int64_t count, Deleter&& deleter) {
        // TODO: check input parameters
        data_owned_ptr_.reset(data, std::forward<Deleter>(deleter));
        data_owned_const_ptr_.reset();
        data_  = data;
        count_ = count;
    }

    template <typename Deleter>
    void reset(const T* data, std::int64_t count, Deleter&& deleter) {
        // TODO: check input parameters
        data_owned_ptr_.reset();
        data_owned_const_ptr_.reset(data, std::forward<Deleter>(deleter));
        data_  = data;
        count_ = count;
    }

    template <typename Y, typename K>
    void reset(const array<Y>& ref, K* data, std::int64_t count) {
        data_owned_ptr_ = detail::shared<T>(ref.data_owned_ptr_, nullptr);
        data_owned_const_ptr_.reset();
        data_           = data;
        count_          = count;
    }

    const T& operator[](std::int64_t index) const {
        return get_data()[index];
    }

    T& operator[](std::int64_t index) {
        return get_mutable_data()[index];
    }

private:
    detail::shared<T> data_owned_ptr_;
    detail::shared<const T> data_owned_const_ptr_;
    std::variant<T*, const T*> data_;

    std::int64_t count_;
};

} // namespace oneapi::dal
