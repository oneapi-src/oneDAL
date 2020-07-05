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

#include "oneapi/dal/detail/memory.hpp"

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
        return empty_impl(host_policy{}, count, detail::host_only_alloc{});
    }

    template <typename K>
    static array<T> full(std::int64_t count, K&& element) {
        return full_impl(host_policy{},
                         count,
                         std::forward<K>(element),
                         detail::host_only_alloc{});
    }

    static array<T> zeros(std::int64_t count) {
        // TODO: can be optimized in future
        return full_impl(host_policy{}, count, T{}, detail::host_only_alloc{});
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    static array<T> empty(sycl::queue& queue,
                          std::int64_t count,
                          sycl::usm::alloc kind = sycl::usm::alloc::shared) {
        return empty_impl(queue, count, kind);
    }

    template <typename K>
    static array<T> full(sycl::queue& queue,
                         std::int64_t count,
                         K&& element,
                         sycl::usm::alloc kind = sycl::usm::alloc::shared) {
        return full_impl(queue, count, std::forward<K>(element), kind);
    }

    static array<T> zeros(sycl::queue& queue,
                          std::int64_t count,
                          sycl::usm::alloc kind = sycl::usm::alloc::shared) {
        // TODO: can be optimized in future
        return full_impl(queue, count, T{}, kind);
    }
#endif

public:
    array() : data_owned_ptr_(nullptr), count_(0) {}

    template <typename Deleter>
    explicit array(T* data, std::int64_t count, Deleter&& deleter) {
        reset(data, count, std::forward<Deleter>(deleter));
    }

    template <typename Y, typename K>
    explicit array(const array<Y>& ref, K* data, std::int64_t count)
            : data_owned_ptr_(ref.data_owned_ptr_, nullptr),
              data_(data),
              count_(count) {}

#ifdef ONEAPI_DAL_DATA_PARALLEL
    explicit array(sycl::queue& queue,
                   T* data,
                   std::int64_t count,
                   const sycl::vector_class<sycl::event>& dependencies = {}) {
        reset(queue, data, count, dependencies);
    }
#endif

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
        return need_mutable_data_impl(host_policy{}, detail::host_only_alloc{});
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    array& need_mutable_data(sycl::queue& queue, sycl::usm::alloc kind = sycl::usm::alloc::shared) {
        return need_mutable_data_impl(queue, kind);
    }
#endif

    std::int64_t get_count() const {
        return count_;
    }

    std::int64_t get_size() const {
        return count_ * sizeof(T);
    }

    void reset() {
        data_owned_ptr_.reset();
        data_  = std::variant<T*, const T*>();
        count_ = 0;
    }

    void reset(std::int64_t count) {
        reset_impl(host_policy{}, count, detail::host_only_alloc{});
    }

    template <typename Deleter>
    void reset(T* data, std::int64_t count, Deleter&& deleter) {
        // TODO: check input parameters
        data_owned_ptr_.reset(data, std::forward<Deleter>(deleter));
        data_  = data_owned_ptr_.get();
        count_ = count;
    }

    template <typename Y, typename K>
    void reset(const array<Y>& ref, K* data, std::int64_t count) {
        data_owned_ptr_ = detail::shared<T>(ref.data_owned_ptr_, nullptr);
        data_           = data;
        count_          = count;
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    void reset(sycl::queue& queue,
               std::int64_t count,
               sycl::usm::alloc kind = sycl::usm::alloc::shared) {
        reset_impl(queue, count, kind);
    }

    void reset(sycl::queue& queue,
               T* data,
               std::int64_t count,
               const sycl::vector_class<sycl::event>& dependencies = {}) {
        reset(data, count, detail::default_delete<T, decltype(queue)>{ queue });
        detail::wait_and_throw(dependencies);
    }
#endif

    const T& operator[](std::int64_t index) const {
        return get_data()[index];
    }

    T& operator[](std::int64_t index) {
        return get_mutable_data()[index];
    }

private:
    template <typename Policy, typename AllocKind>
    static array<T> empty_impl(Policy&& policy, std::int64_t count, AllocKind&& kind) {
        auto data = detail::malloc<T>(policy, count, kind);
        return array<T>{ data, count, detail::default_delete<T, Policy>{ policy } };
    }

    template <typename K, typename Policy, typename AllocKind>
    static array<T> full_impl(Policy&& policy, std::int64_t count, K&& element, AllocKind&& kind) {
        auto array = empty_impl(std::forward<Policy>(policy), count, std::forward<AllocKind>(kind));
        detail::fill(policy, array.get_mutable_data(), count, element);
        return array;
    }

private:
    template <typename Policy, typename AllocKind>
    array& need_mutable_data_impl(Policy&& policy, AllocKind&& kind) {
        if (has_mutable_data() || count_ == 0) {
            return *this;
        }
        else {
            auto immutable_data = get_data();
            auto copy_data      = detail::malloc<T>(policy, count_, kind);
            detail::memcpy(policy, copy_data, immutable_data, sizeof(T) * count_);

            reset(copy_data, count_, detail::default_delete<T, Policy>{ policy });
            return *this;
        }
    }

    template <typename Policy, typename AllocKind>
    void reset_impl(Policy&& policy, std::int64_t count, AllocKind&& kind) {
        auto new_data = detail::malloc<T>(policy, count, kind);
        reset(new_data, count, detail::default_delete<T, Policy>{ policy });
    }

private:
    detail::shared<T> data_owned_ptr_;
    std::variant<T*, const T*> data_;

    std::int64_t count_;
};

} // namespace oneapi::dal
