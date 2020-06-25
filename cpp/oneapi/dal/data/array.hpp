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
#include <algorithm>
#include <cstring>
#include <stdexcept> // TODO: change by onedal exceptions

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/memory.hpp"

namespace oneapi::dal {

template <typename T>
class array {
    static_assert(!std::is_const_v<T>,
                    "array class cannot have const-qualified type of data");

    template <typename U>
    friend class array;

    template <typename Y, typename U>
    friend array<Y> reinterpret_array_cast(const array<U>&);

    template <typename Y, typename U>
    friend array<Y> const_array_cast(const array<U>&);

public:
    using default_delete = std::default_delete<T[]>;
#ifdef ONEAPI_DAL_DATA_PARALLEL
    using dpc_default_delete = detail::dpc_default_delete<T>;
#endif

public:
    template <typename K>
    static array<T> full(std::int64_t count, K&& element) {
        return full_impl(detail::host_policy{},
            count, std::forward<K>(element), detail::host_only_alloc{});
    }

    static array<T> zeros(std::int64_t count) {
        auto* data = new T[count];
        std::memset(data, 0, sizeof(T)*count);
        return array<T> { data, count, default_delete{} };
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    template <typename K>
    static array<T> full(sycl::queue queue,
                         std::int64_t count, K&& element,
                         sycl::usm::alloc kind = sycl::usm::alloc::shared) {
        return full_impl(queue, count, std::forward<K>(element), kind);
    }

    static array<T> zeros(sycl::queue queue,
                          std::int64_t count,
                          sycl::usm::alloc kind = sycl::usm::alloc::shared) {
        auto device = queue.get_device();
        auto context = queue.get_context();
        auto* data = sycl::malloc<T>(count, device, context, kind);
        auto event = queue.memset(data, 0, sizeof(T)*count);

        return array<T> { queue, data, count, {event} };
    }
#endif

public:
    array()
        : data_owned_ptr_(nullptr),
          count_(0),
          capacity_(0) {}

    explicit array(std::int64_t count)
        : array() {
        reset(count);
    }

    template <typename U = T*,
              typename = std::enable_if_t<std::is_pointer_v<U>>>
    explicit array(U data, std::int64_t count)
        : data_owned_ptr_(nullptr),
          data_(data),
          count_(count),
          capacity_(0) {}

    template <typename Deleter>
    explicit array(T* data, std::int64_t count, Deleter&& deleter)
        : array() {
        reset(data, count, std::forward<Deleter>(deleter));
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    explicit array(sycl::queue queue,
                   std::int64_t count,
                   sycl::usm::alloc kind = sycl::usm::alloc::shared)
        : array() {
        reset(queue, count, kind);
    }

    explicit array(sycl::queue queue,
                   T* data, std::int64_t count,
                   sycl::vector_class<sycl::event> dependencies = {})
        : array() {
            reset(queue, data, count, dependencies);
        }
#endif

    T* get_mutable_data() const {
        return std::get<T*>(data_); // TODO: convert to dal exception
    }

    const T* get_data() const {
        if (auto ptr_val = std::get_if<T*>(&data_)) {
            return *ptr_val;
        } else {
            return std::get<const T*>(data_);
        }
    }

    bool has_mutable_data() const {
        return std::holds_alternative<T*>(data_) && (get_mutable_data() != nullptr);
    }

    array& unique() {
        if (is_data_owner() || count_ == 0) {
            return *this;
        } else {
            auto immutable_data = get_data();
            auto copy_data = new T[count_];

            for (std::int64_t i = 0; i < count_; i++) {
                copy_data[i] = immutable_data[i];
            }

            reset(copy_data, count_, default_delete{});
            return *this;
        }
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    array& unique(sycl::queue queue,
                  sycl::usm::alloc kind = sycl::usm::alloc::shared) {
        if (is_data_owner() || count_ == 0) {
            return *this;
        } else {
            auto immutable_data = get_data();
            auto device = queue.get_device();
            auto context = queue.get_context();
            auto* copy_data = sycl::malloc<T>(count_, device, context, kind);
            auto event = queue.memcpy(copy_data, immutable_data, sizeof(T)*count_);

            reset(queue, copy_data, count_, {event});
            return *this;
        }
    }
#endif

    std::int64_t get_count() const {
        return count_;
    }

    std::int64_t get_size() const {
        return count_ * sizeof(T);
    }

    std::int64_t get_capacity() const {
        return capacity_;
    }

    bool is_data_owner() const {
        if (data_owned_ptr_ == nullptr) {
            return false;
        } else if (auto ptr_val = std::get_if<T*>(&data_)) {
            return *ptr_val == data_owned_ptr_.get();
        } else if (auto ptr_val = std::get_if<const T*>(&data_)) {
            return *ptr_val == data_owned_ptr_.get();
        } else {
            return false;
        }
    }

    void reset() {
        data_owned_ptr_.reset();
        data_ = std::variant<T*, const T*>();
        count_ = 0;
        capacity_ = 0;
    }

    void reset(std::int64_t count) {
        reset(new T[count], count, default_delete{});
    }

    template <typename Deleter>
    void reset(T* data, std::int64_t count, Deleter&& deleter) {
        // TODO: check input parameters
        data_owned_ptr_.reset(data, std::forward<Deleter>(deleter));
        data_ = data_owned_ptr_.get();
        count_ = count;
        capacity_ = count;
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    void reset(sycl::queue queue,
               std::int64_t count,
               sycl::usm::alloc kind = sycl::usm::alloc::shared) {
        auto device = queue.get_device();
        auto context = queue.get_context();
        auto* new_data = sycl::malloc<T>(count, device, context, kind);

        reset(new_data, count, dpc_default_delete{ queue });
    }

    void reset(sycl::queue queue,
               T* data, std::int64_t count,
               sycl::vector_class<sycl::event> dependencies = {}) {
        reset(data, count, dpc_default_delete{ queue });
        for (auto& event : dependencies) {
            event.wait();
        }
    }
#endif

    template <typename U = T*>
    void reset_not_owning(U data = nullptr, std::int64_t count = 0) {
        data_ = data;
        count_ = count;
    }

    void resize(std::int64_t count) {
        if (is_data_owner() == false) {
            throw std::runtime_error("cannot resize array with non-owning data");
        } else if (count <= 0) {
            reset_not_owning();
        } else if (get_capacity() < count) {
            T* new_data = new T[count];
            std::int64_t min_count = std::min(count, get_count());

            for (std::int64_t i = 0; i < min_count; i++) {
                new_data[i] = (*this)[i];
            }

            try {
                reset(new_data, count, default_delete{});
            } catch (const std::exception&) {
                delete[] new_data;
                throw;
            }

        } else {
            count_ = count;
        }
    }

#ifdef ONEAPI_DAL_DATA_PARALLEL
    void resize(sycl::queue queue,
                std::int64_t count,
                sycl::usm::alloc kind = sycl::usm::alloc::shared) {
        if (is_data_owner() == false) {
            throw std::runtime_error("cannot resize array with non-owning data");
        } else if (count <= 0) {
            reset_not_owning();
        } else if (get_capacity() < count) {
            auto device = queue.get_device();
            auto context = queue.get_context();
            auto* new_data = sycl::malloc<T>(count, device, context, kind);

            std::int64_t min_count = std::min(count, get_count());
            auto event = queue.memcpy(new_data, this->get_data(), sizeof(T)*min_count);

            try {
                reset(queue, new_data, count, {event});
            } catch (const std::exception&) {
                sycl::free(new_data, queue);
                throw;
            }

        } else {
            count_ = count;
        }
    }
#endif

    const T& operator [](std::int64_t index) const {
        return get_data()[index];
    }

    T& operator [](std::int64_t index) {
        return get_mutable_data()[index];
    }
private:
    template <typename K, typename Policy, typename AllocKind>
    static array<T> full_impl(Policy&& policy,
                              std::int64_t count, K&& element,
                              AllocKind&& kind) {
        auto* data = detail::malloc<T>(policy, count, kind);
        detail::fill(policy, data, count, element);
        return array<T> { data, count, [policy](T* pointer) { detail::free(policy, pointer); } };
    }

private:
    detail::shared<T> data_owned_ptr_;
    std::variant<T*, const T*> data_;

    std::int64_t count_;
    std::int64_t capacity_;
};

} // namespace oneapi::dal
