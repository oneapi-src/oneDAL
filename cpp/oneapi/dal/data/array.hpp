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
#include "oneapi/dal/detail/common.hpp"

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

public:
    template <typename K>
    static array<T> full(std::int64_t element_count, K&& element) {
        auto* data = new T[element_count];

        for (std::int64_t i = 0; i < element_count; i++) {
            data[i] = std::forward<K>(element);
        }

        return array<T> { data, element_count, default_delete{} };
    }

    static array<T> zeros(std::int64_t element_count) {
        auto* data = new T[element_count];
        std::memset(data, 0, sizeof(T)*element_count);
        return array<T> { data, element_count, default_delete{} };
    }

public:
    array()
        : data_owned_ptr_(nullptr),
          size_(0),
          capacity_(0) {}

    template <typename U = T*>
    explicit array(U data, std::int64_t size)
        : data_owned_ptr_(nullptr),
          data_(data),
          size_(size),
          capacity_(0) {}

    template <typename Deleter>
    explicit array(T* data, std::int64_t size, Deleter&& deleter)
        : data_owned_ptr_(data, std::forward<Deleter>(deleter)),
            data_(data),
            size_(size),
            capacity_(size) {}

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
        if (is_data_owner() || size_ == 0) {
            return *this;
        } else {
            auto immutable_data = get_data();
            auto copy_data = new T[size_];

            for (std::int64_t i = 0; i < size_; i++) {
                copy_data[i] = immutable_data[i];
            }

            reset(copy_data, size_, default_delete{});
            return *this;
        }
    }

    std::int64_t get_size() const {
        return size_;
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
        size_ = 0;
        capacity_ = 0;
    }

    void reset(std::int64_t size) {
        data_owned_ptr_.reset(new T[size], default_delete{});
        data_ = data_owned_ptr_.get();
        size_ = size;
        capacity_ = size;
    }

    template <typename Deleter>
    void reset(T* data, std::int64_t size, Deleter&& deleter) {
        // TODO: check input parameters
        data_owned_ptr_.reset(data, std::forward<Deleter>(deleter));
        data_ = data_owned_ptr_.get();
        size_ = size;
        capacity_ = size;
    }

    template <typename U = T*>
    void reset_not_owning(U data = nullptr, std::int64_t size = 0) {
        data_ = data;
        size_ = size;
    }

    void resize(std::int64_t size) {
        if (is_data_owner() == false) {
            throw std::runtime_error("cannot resize array with non-owning data");
        } else if (size <= 0) {
            reset_not_owning();
        } else if (get_capacity() < size) {
            T* new_data = new T[size];
            std::int64_t min_size = std::min(size, get_size());

            for (std::int64_t i = 0; i < min_size; i++) {
                new_data[i] = (*this)[i];
            }

            try {
                reset(new_data, size, default_delete{});
            } catch (const std::exception&) {
                delete[] new_data;
                throw;
            }

        } else {
            size_ = size;
        }
    }

    const T& operator [](std::int64_t index) const {
        return get_data()[index];
    }

    T& operator [](std::int64_t index) {
        return get_mutable_data()[index];
    }

private:
    detail::shared<T> data_owned_ptr_;
    std::variant<T*, const T*> data_;

    std::int64_t size_;
    std::int64_t capacity_;
};

} // namespace oneapi::dal
