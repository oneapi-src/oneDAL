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

namespace oneapi::dal {
namespace v1 {

// This class is needed for compatibility with the oneDAL 2021.1.
// This should be removed in 2022.1.
template <typename T>
class array {
    static_assert(!std::is_const_v<T>, "array class cannot have const-qualified type of data");

    template <typename U>
    friend class array;

    using impl_t = detail::v1::array_impl<T>;

public:
    using data_t = T;

    static array<T> empty(std::int64_t count) {
        return array<T>{
            impl_t::empty(detail::default_host_policy{}, count, detail::host_allocator<T>())
        };
    }

#ifdef ONEDAL_DATA_PARALLEL
    static array<T> empty(const sycl::queue& queue,
                          std::int64_t count,
                          const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) {
        return array<T>{ impl_t::empty(detail::data_parallel_policy{ queue },
                                       count,
                                       detail::data_parallel_allocator<T>(queue, alloc)) };
    }
#endif

    template <typename K>
    static array<T> full(std::int64_t count, K&& element) {
        return array<T>{ impl_t::full(detail::default_host_policy{},
                                      count,
                                      std::forward<K>(element),
                                      detail::host_allocator<T>()) };
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename K>
    static array<T> full(sycl::queue& queue,
                         std::int64_t count,
                         K&& element,
                         const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) {
        return array<T>{ impl_t::full(detail::data_parallel_policy{ queue },
                                      count,
                                      std::forward<K>(element),
                                      detail::data_parallel_allocator<T>(queue, alloc)) };
    }
#endif

    static array<T> zeros(std::int64_t count) {
        // TODO: can be optimized in future
        return array<T>{
            impl_t::full(detail::default_host_policy{}, count, T{}, detail::host_allocator<T>())
        };
    }

#ifdef ONEDAL_DATA_PARALLEL
    static array<T> zeros(sycl::queue& queue,
                          std::int64_t count,
                          const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) {
        return array<T>{ impl_t::full(dal::detail::data_parallel_policy{ queue },
                                      count,
                                      T{},
                                      detail::data_parallel_allocator<T>(queue, alloc)) };
    }
#endif

    template <typename Y>
    static array<T> wrap(Y* data, std::int64_t count) {
        return array<T>{ data, count, dal::detail::empty_delete<const T>{} };
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename Y>
    static array<T> wrap(Y* data,
                         std::int64_t count,
                         const sycl::vector_class<sycl::event>& dependencies) {
        return array<T>{ data, count, dal::detail::empty_delete<const T>{}, dependencies };
    }
#endif

    array() : impl_(new impl_t()) {
        reset_data();
    }

    array(const array<T>& other) : impl_(new impl_t(*other.impl_)) {
        update_data(impl_.get());
    }

    array(array<T>&& other) : impl_(std::move(other.impl_)) {
        update_data(impl_.get());
        other.reset_data();
    }

    template <typename Deleter>
    explicit array(T* data, std::int64_t count, Deleter&& deleter)
            : impl_(new impl_t(data, count, std::forward<Deleter>(deleter))) {
        update_data(data, count);
    }

    template <typename ConstDeleter>
    explicit array(const T* data, std::int64_t count, ConstDeleter&& deleter)
            : impl_(new impl_t(data, count, std::forward<ConstDeleter>(deleter))) {
        update_data(data, count);
    }

    explicit array(const std::shared_ptr<T>& data, std::int64_t count)
            : impl_(new impl_t(data, count)) {
        update_data(data.get(), count);
    }

    explicit array(const std::shared_ptr<const T>& data, std::int64_t count)
            : impl_(new impl_t(data, count)) {
        update_data(data.get(), count);
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename Deleter>
    explicit array(const sycl::queue& queue,
                   T* data,
                   std::int64_t count,
                   Deleter&& deleter,
                   const sycl::vector_class<sycl::event>& dependencies = {})
            : impl_(new impl_t(data, count, std::forward<Deleter>(deleter))) {
        update_data(impl_.get());
        sycl::event::wait_and_throw(dependencies);
    }

    template <typename ConstDeleter>
    explicit array(const sycl::queue& queue,
                   const T* data,
                   std::int64_t count,
                   ConstDeleter&& deleter,
                   const sycl::vector_class<sycl::event>& dependencies = {})
            : impl_(new impl_t(data, count, std::forward<ConstDeleter>(deleter))) {
        update_data(impl_.get());
        sycl::event::wait_and_throw(dependencies);
    }
#endif

    template <typename Y, typename K>
    explicit array(const array<Y>& ref, K* data, std::int64_t count)
            : impl_(new impl_t(*ref.impl_, data, count)) {
        update_data(impl_.get());
    }

    array<T> operator=(const array<T>& other) {
        array<T> tmp{ other };
        swap(*this, tmp);
        return *this;
    }

    array<T> operator=(array<T>&& other) {
        swap(*this, other);
        return *this;
    }

    T* get_mutable_data() const {
        if (!has_mutable_data()) {
            throw domain_error(dal::detail::error_messages::array_does_not_contain_mutable_data());
        }
        return mutable_data_ptr_;
    }

    const T* get_data() const noexcept {
        return data_ptr_;
    }

    bool has_mutable_data() const noexcept {
        return mutable_data_ptr_ != nullptr;
    }

    array& need_mutable_data() {
        impl_->need_mutable_data(detail::default_host_policy{}, detail::host_allocator<data_t>{});
        update_data(impl_->get_mutable_data(), impl_->get_count());
        return *this;
    }

#ifdef ONEDAL_DATA_PARALLEL
    array& need_mutable_data(sycl::queue& queue,
                             const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) {
        impl_->need_mutable_data(detail::data_parallel_policy{ queue },
                                 detail::data_parallel_allocator<T>(queue, alloc));
        update_data(impl_->get_mutable_data(), impl_->get_count());
        return *this;
    }
#endif

    std::int64_t get_count() const noexcept {
        return count_;
    }

    std::int64_t get_size() const noexcept {
        return count_ * sizeof(T);
    }

    void reset() {
        impl_->reset();
        reset_data();
    }

    void reset(std::int64_t count) {
        impl_->reset(detail::default_host_policy{}, count, detail::host_allocator<data_t>{});
        update_data(impl_->get_mutable_data(), count);
    }

#ifdef ONEDAL_DATA_PARALLEL
    void reset(const sycl::queue& queue,
               std::int64_t count,
               const sycl::usm::alloc& alloc = sycl::usm::alloc::shared) {
        impl_->reset(detail::data_parallel_policy{ queue },
                     count,
                     detail::data_parallel_allocator<T>(queue, alloc));
        update_data(impl_->get_mutable_data(), count);
    }
#endif

    template <typename Deleter>
    void reset(T* data, std::int64_t count, Deleter&& deleter) {
        impl_->reset(data, count, std::forward<Deleter>(deleter));
        update_data(data, count);
    }

    template <typename ConstDeleter>
    void reset(const T* data, std::int64_t count, ConstDeleter&& deleter) {
        impl_->reset(data, count, std::forward<ConstDeleter>(deleter));
        update_data(data, count);
    }

#ifdef ONEDAL_DATA_PARALLEL
    template <typename Y, typename YDeleter>
    void reset(Y* data,
               std::int64_t count,
               YDeleter&& deleter,
               const sycl::vector_class<sycl::event>& dependencies) {
        sycl::event::wait_and_throw(dependencies);
        reset(data, count, std::forward<YDeleter>(deleter));
    }
#endif

    template <typename Y>
    void reset(const array<Y>& ref, T* data, std::int64_t count) {
        impl_->reset(*ref.impl_, data, count);
        update_data(data, count);
    }

    template <typename Y>
    void reset(const array<Y>& ref, const T* data, std::int64_t count) {
        impl_->reset(*ref.impl_, data, count);
        update_data(data, count);
    }

    const T& operator[](std::int64_t index) const noexcept {
        return data_ptr_[index];
    }

    template <typename Y>
    void reset(const v2::array<Y>& ref) {
        auto& impl_v2 = detail::get_impl(ref);
        if (ref.has_mutable_data()) {
            impl_->reset(impl_v2.get_shared(), ref.get_count());
        }
        else {
            impl_->reset(impl_v2.get_cshared(), ref.get_count());
        }
        update_data(impl_.get());
    }

    v2::array<T> v2() const {
        if (has_mutable_data()) {
            return v2::array<T>{ impl_->get_shared(), count_ };
        }
        else {
            return v2::array<T>{ impl_->get_cshared(), count_ };
        }
    }

private:
    static void swap(array<T>& a, array<T>& b) {
        std::swap(a.impl_, b.impl_);
        std::swap(a.data_ptr_, b.data_ptr_);
        std::swap(a.mutable_data_ptr_, b.mutable_data_ptr_);
        std::swap(a.count_, b.count_);
    }

    array(impl_t* impl) : impl_(impl) {
        update_data(impl_.get());
    }

    void update_data(impl_t* impl) {
        if (impl->has_mutable_data()) {
            update_data(impl->get_mutable_data(), impl->get_count());
        }
        else {
            update_data(impl->get_data(), impl->get_count());
        }
    }

    void update_data(const T* data, std::int64_t count) noexcept {
        data_ptr_ = data;
        mutable_data_ptr_ = nullptr;
        count_ = count;
    }

    void update_data(T* data, std::int64_t count) noexcept {
        data_ptr_ = data;
        mutable_data_ptr_ = data;
        count_ = count;
    }

    void reset_data() noexcept {
        data_ptr_ = nullptr;
        mutable_data_ptr_ = nullptr;
        count_ = 0;
    }

private:
    detail::unique<impl_t> impl_;
    const T* data_ptr_;
    T* mutable_data_ptr_;
    std::int64_t count_;
};

} // namespace v1
} // namespace oneapi::dal
