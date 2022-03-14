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

namespace oneapi::dal::detail {
namespace v1 {

template <typename T>
class paged_vector_page {
public:
    explicit paged_vector_page(std::int64_t min_page_size)
            : next_(nullptr),
              data_(nullptr),
              count_(0),
              capacity_(min_page_size) {
        ONEDAL_ASSERT(min_page_size > 0);
    }

    ~paged_vector_page() {
        free(default_host_policy{}, data_);
        next_ = nullptr;
        data_ = nullptr;
        count_ = 0;
        capacity_ = 0;
    }

    paged_vector_page(const paged_vector_page&) = delete;
    paged_vector_page& operator=(const paged_vector_page&) = delete;

    paged_vector_page* make_next(std::int64_t min_page_size) {
        return next_ = new paged_vector_page{ min_page_size };
    }

    paged_vector_page* get_next() const {
        return next_;
    }

    std::int64_t get_capacity() const {
        return capacity_;
    }

    std::int64_t get_count() const {
        return count_;
    }

    const T* get_data() const {
        return data_;
    }

    bool try_write(const T* data, std::int64_t count) {
        if (count_ == 0) {
            ONEDAL_ASSERT(data_ == nullptr);
            capacity_ = std::max(capacity_, count);
            data_ = malloc<T>(default_host_policy{}, capacity_);
        }

        ONEDAL_ASSERT_SUM_OVERFLOW(std::int64_t, count_, count);
        if (count_ + count > capacity_) {
            return false;
        }

        for (std::int64_t i = 0; i < count; i++) {
            data_[count_ + i] = data[i];
        }

        count_ += count;
        ONEDAL_ASSERT(count_ <= capacity_);

        return true;
    }

private:
    paged_vector_page* next_;
    T* data_;
    std::int64_t count_;
    std::int64_t capacity_;
};

template <typename T>
class paged_vector : public base {
public:
    using page_t = paged_vector_page<T>;

    explicit paged_vector(std::int64_t min_page_size) {
        if (min_page_size <= 0) {
            throw invalid_argument{ error_messages::page_size_leq_zero() };
        }

        init(min_page_size);
    }

    ~paged_vector() {
        destroy();
    }

    paged_vector(const paged_vector&) = delete;
    paged_vector& operator=(const paged_vector&) = delete;

    void push_back(const T* data, std::int64_t count) {
        ONEDAL_ASSERT_SUM_OVERFLOW(std::int64_t, page_count_, 1);
        ONEDAL_ASSERT_SUM_OVERFLOW(std::int64_t, total_count_, count);

        while (!last_->try_write(data, count)) {
            const std::int64_t page_size = std::max(min_page_size_, count);
            last_ = last_->make_next(page_size);
            page_count_++;
        }

        total_count_ += count;
    }

    std::int64_t get_page_count() const {
        return page_count_;
    }

    std::int64_t get_count() const {
        return total_count_;
    }

    array<T> to_array() const {
        if (total_count_ == 0) {
            return array<T>{};
        }

        auto result = array<T>::empty(total_count_);
        T* result_ptr = result.get_mutable_data();

        std::int64_t element_counter = 0;
        iterate_over_pages([&](page_t* page) {
            if (page->get_count() > 0) {
                memcpy(default_host_policy{},
                       result_ptr + element_counter,
                       page->get_data(),
                       page->get_count());
            }
            element_counter += page->get_count();
            ONEDAL_ASSERT(element_counter <= total_count_);
        });

        return result;
    }

    void reset() {
        destroy();
        init(min_page_size_);
    }

private:
    void init(std::int64_t min_page_size) {
        head_ = new page_t{ min_page_size };
        last_ = head_;
        page_count_ = 1;
        total_count_ = 0;
        min_page_size_ = min_page_size;
    }

    void destroy() {
        iterate_over_pages([&](page_t* page) {
            delete page;
        });

        head_ = nullptr;
        last_ = nullptr;
        page_count_ = 0;
        total_count_ = 0;
    }

    template <typename Body>
    void iterate_over_pages(Body&& body) const {
        page_t* current = head_;
        while (current) {
            page_t* tmp = current;
            current = current->get_next();
            body(tmp);
        }
    }

    page_t* head_;
    std::int64_t page_count_;
    std::int64_t total_count_;
    std::int64_t min_page_size_;
    page_t* last_;
};

} // namespace v1

using v1::paged_vector_page;
using v1::paged_vector;

} // namespace oneapi::dal::detail
