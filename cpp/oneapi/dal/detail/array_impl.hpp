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

#include <variant>
#include <optional>

#include "oneapi/dal/detail/memory.hpp"
#include "oneapi/dal/detail/policy.hpp"
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
        return new array_impl<T>{ policy, data, count, [alloc, count](T* ptr) {
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

    array_impl() : count_(0) {}

    template <typename Policy>
    array_impl(const Policy& policy, const shared& data, std::int64_t count) {
        reset(policy, data, count);
    }

    template <typename Policy>
    array_impl(const Policy& policy, const cshared& data, std::int64_t count) {
        reset(policy, data, count);
    }

    template <typename Policy, typename Deleter>
    array_impl(const Policy& policy, T* data, std::int64_t count, Deleter&& d) {
        reset(policy, data, count, std::forward<Deleter>(d));
    }

    template <typename Policy, typename ConstDeleter>
    array_impl(const Policy& policy, const T* data, std::int64_t count, ConstDeleter&& d) {
        reset(policy, data, count, std::forward<ConstDeleter>(d));
    }

    template <typename Y, typename K>
    array_impl(const array_impl<Y>& ref, K* data, std::int64_t count) {
        reset(ref, data, count);
    }

    std::int64_t get_count() const noexcept {
        return count_;
    }

    const T* get_data() const noexcept {
        if (const auto& mut_ptr = get_if_shared()) {
            return mut_ptr->get();
        }
        else {
            const auto& immut_ptr = get_cshared();
            return immut_ptr.get();
        }
    }

    T* get_mutable_data() const {
        if (const auto& mut_ptr = get_if_shared()) {
            return mut_ptr->get();
        }
        else {
            throw internal_error(
                dal::detail::error_messages::array_does_not_contain_mutable_data());
        }
    }

    bool has_mutable_data() const noexcept {
        return std::holds_alternative<shared>(data_owned_) && (get_mutable_data() != nullptr);
    }

    void need_mutable_data() {
        if (has_mutable_data() || count_ == 0) {
            return;
        }
        else {
            data_owned_ = copy();
        }
    }

    void reset() {
        data_owned_ = std::variant<cshared, shared>();
        count_ = 0;
        reset_policy();
    }

    template <typename Policy, typename Allocator>
    void reset(const Policy& policy, std::int64_t count, const Allocator& alloc) {
        auto new_data = alloc.allocate(count);
        reset(policy, new_data, count, [alloc, count](T* ptr) {
            alloc.deallocate(ptr, count);
        });
    }

    template <typename Policy>
    void reset(const Policy& policy, const shared& data, std::int64_t count) {
        data_owned_ = data;
        count_ = count;
        reset_policy(policy);
    }

    template <typename Policy>
    void reset(const Policy& policy, const cshared& data, std::int64_t count) {
        data_owned_ = data;
        count_ = count;
        reset_policy(policy);
    }

    template <typename Policy, typename Deleter>
    void reset(const Policy& policy, T* data, std::int64_t count, Deleter&& deleter) {
        data_owned_ = shared(data, std::forward<Deleter>(deleter));
        count_ = count;
        reset_policy(policy);
    }

    template <typename Policy, typename ConstDeleter>
    void reset(const Policy& policy, const T* data, std::int64_t count, ConstDeleter&& deleter) {
        data_owned_ = cshared(data, std::forward<ConstDeleter>(deleter));
        count_ = count;
        reset_policy(policy);
    }

    template <typename Y>
    void reset(const array_impl<Y>& ref, T* data, std::int64_t count) {
        check_array_has_ownership_structure(ref);
        if (ref.has_mutable_data()) {
            using shared_y = typename array_impl<Y>::shared;
            if (const auto& ptr = std::get_if<shared_y>(&ref.data_owned_)) {
                data_owned_ = shared(*ptr, data);
            }
        }
        else {
            data_owned_ = shared(ref.get_cshared(), data);
        }
        count_ = count;
        reset_policy(ref);
    }

    template <typename Y>
    void reset(const array_impl<Y>& ref, const T* data, std::int64_t count) {
        check_array_has_ownership_structure(ref);
        if (ref.has_mutable_data()) {
            using shared_y = typename array_impl<Y>::shared;
            if (const auto& ptr = std::get_if<shared_y>(&ref.data_owned_)) {
                data_owned_ = cshared(*ptr, data);
            }
        }
        else {
            data_owned_ = cshared(ref.get_cshared(), data);
        }
        count_ = count;
        reset_policy(ref);
    }

#ifdef ONEDAL_DATA_PARALLEL
    std::optional<data_parallel_policy> get_policy() const {
        return dp_policy_;
    }
#endif

private:
    void reset_policy() {
#ifdef ONEDAL_DATA_PARALLEL
        dp_policy_.reset();
#endif
    }

    template <typename Policy>
    void reset_policy(const Policy& policy) {
#ifdef ONEDAL_DATA_PARALLEL
        if constexpr (is_data_parallel_policy_v<Policy>) {
            dp_policy_ = policy;
        }
#endif
    }

    template <typename Y>
    void reset_policy(const array_impl<Y>& ref) {
#ifdef ONEDAL_DATA_PARALLEL
        dp_policy_ = ref.dp_policy_;
#endif
    }

    template <typename Y>
    void check_array_has_ownership_structure(const array_impl<Y>& ref) {
        if (ref.get_if_cshared() == nullptr && ref.get_if_shared() == nullptr) {
            throw internal_error(
                dal::detail::error_messages::array_does_not_contain_ownership_structure());
        }
    }

    shared copy() {
#ifdef ONEDAL_DATA_PARALLEL
        if (dp_policy_.has_value()) {
            const auto policy = dp_policy_.value();
            const auto context = policy.get_queue().get_context();
            const auto alloc_kind = sycl::get_pointer_type(get_data(), context);
            const auto allocator = data_parallel_allocator<T>{ policy, alloc_kind };
            return copy_generic(policy, allocator);
        }
        else {
            return copy_generic(default_host_policy{}, host_allocator<T>{});
        }
#else
        return copy_generic(default_host_policy{}, host_allocator<T>{});
#endif
    }

    template <typename Policy, typename Allocator>
    shared copy_generic(const Policy& policy, const Allocator& alloc) {
        const T* data = get_data();
        T* data_copy = alloc.allocate(count_);
        memcpy(policy, data_copy, data, sizeof(T) * count_);
        return shared(data_copy, [alloc, count = this->count_](T* ptr) {
            alloc.deallocate(ptr, count);
        });
    }

    shared get_shared() const {
        return std::get<shared>(data_owned_);
    }

    const shared* get_if_shared() const noexcept {
        return std::get_if<shared>(&data_owned_);
    }

    cshared get_cshared() const {
        return std::get<cshared>(data_owned_);
    }

    const cshared* get_if_cshared() const noexcept {
        return std::get_if<cshared>(&data_owned_);
    }

    std::variant<cshared, shared> data_owned_;
    std::int64_t count_;
#ifdef ONEDAL_DATA_PARALLEL
    std::optional<data_parallel_policy> dp_policy_;
#endif
};

} // namespace v1

using v1::array_impl;

} // namespace oneapi::dal::detail
