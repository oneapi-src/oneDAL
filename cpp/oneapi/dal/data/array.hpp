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
    static array<T> empty(std::int64_t count) { // support allocators in private (empty_impl)
        return empty_impl(detail::default_host_policy{}, count, detail::host_allocator<data_t>{});
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    static array<T> empty(
        const sycl::queue& queue,
        std::int64_t count,
        const sycl::usm::alloc& alloc =
            sycl::usm::alloc::shared) { // create our allocators from queue and kind
        return empty_impl(detail::data_parallel_policy{ queue },
                          count,
                          detail::data_parallel_allocator<T>(queue, alloc));
    }
#endif

    template <typename K>
    static array<T> full(std::int64_t count, K&& element) {
        return full_impl(detail::default_host_policy{},
                         count,
                         std::forward<K>(element),
                         detail::host_allocator<data_t>{});
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    template <typename K>
    static array<T> full(sycl::queue& queue,
                         std::int64_t count,
                         K&& element,
                         const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) {
        return full_impl(detail::data_parallel_policy{ queue },
                         count,
                         std::forward<K>(element),
                         detail::data_parallel_allocator<T>(queue, alloc));
    }
#endif

    static array<T> zeros(std::int64_t count) {
        // TODO: can be optimized in future
        return full_impl(detail::default_host_policy{},
                         count,
                         T{},
                         detail::host_allocator<data_t>{});
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    static array<T> zeros(sycl::queue& queue,
                          std::int64_t count,
                          const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) {
        // TODO: can be optimized in future
        return full_impl(detail::data_parallel_policy{ queue },
                         count,
                         T{},
                         detail::data_parallel_allocator<T>(queue, alloc));
    }
#endif
    template <typename Y>
    static array<T> wrap(Y* data, std::int64_t count) {
        return array<T> { data, count, empty_delete<const T>{} };
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    template <typename Y>
    static array<T> wrap(Y* data,
                         std::int64_t count,
                         const sycl::vector_class<sycl::event>& dependencies) {
        return array<T> { data, count, empty_delete<const T>{}, dependencies };
    }
#endif

public:
    array() : data_owned_ptr_(nullptr), data_owned_const_ptr_(nullptr), count_(0) {}

    template <typename Deleter>
    explicit array(T* data, std::int64_t count, Deleter&& deleter) {
        reset(data, count, std::forward<Deleter>(deleter));
    }

    template <typename ConstDeleter>
    explicit array(const T* data, std::int64_t count, ConstDeleter&& deleter) {
        // TODO: static assert on deleter argument
        reset(data, count, std::forward<ConstDeleter>(deleter));
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    template <typename Deleter>
    explicit array(T* data,
          std::int64_t count,
          Deleter&& deleter,
          const sycl::vector_class<sycl::event>& dependencies) {
        reset(data, count, std::forward<Deleter>(deleter), dependencies);
    }

    template <typename ConstDeleter>
    explicit array(const T* data,
          std::int64_t count,
          ConstDeleter&& deleter,
          const sycl::vector_class<sycl::event>& dependencies) {
        reset(data, count, std::forward<ConstDeleter>(deleter), dependencies);
    }
#endif

    template <typename Y, typename K>
    explicit array(const array<Y>& ref, K* data, std::int64_t count)
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
        return need_mutable_data_impl(detail::default_host_policy{},
                                      detail::host_allocator<data_t>{});
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    array& need_mutable_data(sycl::queue& queue,
                             const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) {
        return need_mutable_data_impl(detail::data_parallel_policy{ queue },
                                      detail::data_parallel_allocator<T>(queue, alloc));
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
        data_owned_const_ptr_.reset();
        data_  = std::variant<T*, const T*>();
        count_ = 0;
    }

    void reset(std::int64_t count) {
        reset_impl(detail::default_host_policy{}, count, detail::host_allocator<data_t>{});
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    void reset(const sycl::queue& queue,
               std::int64_t count,
               const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) {
        reset_impl(detail::data_parallel_policy{ queue },
                   count,
                   detail::data_parallel_allocator<T>(queue, alloc));
    }
#endif

    template <typename Deleter>
    void reset(T* data, std::int64_t count, Deleter&& deleter) {
        // TODO: check input parameters
        data_owned_ptr_.reset(data, std::forward<Deleter>(deleter));
        data_owned_const_ptr_.reset();
        data_  = data;
        count_ = count;
    }

    template <typename ConstDeleter>
    void reset(const T* data, std::int64_t count, ConstDeleter&& deleter) {
        // TODO: check input parameters
        data_owned_ptr_.reset();
        data_owned_const_ptr_.reset(data, std::forward<ConstDeleter>(deleter));
        data_  = data;
        count_ = count;
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    template <typename Y, typename YDeleter>
    void reset(Y* data,
               std::int64_t count,
               YDeleter&& deleter,
               const sycl::vector_class<sycl::event>& dependencies) {
        detail::wait_and_throw(dependencies);
        reset(data, count, std::forward<YDeleter>(deleter));
    }
#endif

    template <typename Y, typename K>
    void reset(const array<Y>& ref, K* data, std::int64_t count) {
        data_owned_ptr_ = detail::shared<T>(ref.data_owned_ptr_, nullptr);
        data_owned_const_ptr_.reset();
        data_  = data;
        count_ = count;
    }

    const T& operator[](std::int64_t index) const {
        return get_data()[index];
    }

    T& operator[](std::int64_t index) {
        return get_mutable_data()[index];
    }

private:
    template <typename Policy, typename Allocator>
    static array<T> empty_impl(const Policy& policy, std::int64_t count, const Allocator& alloc) {
        auto data = alloc.allocate(count);
        return array<T>{ data, count, [alloc, count](T* ptr) {
                            alloc.deallocate(ptr, count);
                        } };
    }

    template <typename Policy, typename K, typename Allocator>
    static array<T> full_impl(const Policy& policy,
                              std::int64_t count,
                              K&& element,
                              const Allocator& alloc) {
        auto array = empty_impl(policy, count, alloc);
        detail::fill(policy, array.get_mutable_data(), count, std::forward<K>(element));
        return array;
    }

private:
    template <typename Policy, typename Allocator>
    array& need_mutable_data_impl(const Policy& policy, const Allocator& alloc) {
        if (has_mutable_data() || count_ == 0) {
            return *this;
        }
        else {
            auto immutable_data = get_data();
            auto copy_data      = alloc.allocate(count_);
            detail::memcpy(policy, copy_data, immutable_data, sizeof(T) * count_);

            reset(copy_data, count_, [alloc, count = this->count_](T* ptr) {
                alloc.deallocate(ptr, count);
            });
            return *this;
        }
    }

    template <typename Policy, typename Allocator>
    void reset_impl(const Policy& policy, std::int64_t count, const Allocator& alloc) {
        auto new_data = alloc.allocate(count);
        reset(new_data, count, [alloc, count](T* ptr) {
            alloc.deallocate(ptr, count);
        });
    }

private:
    detail::shared<T> data_owned_ptr_;
    detail::shared<const T> data_owned_const_ptr_;
    std::variant<T*, const T*> data_;

    std::int64_t count_;
};

} // namespace oneapi::dal
