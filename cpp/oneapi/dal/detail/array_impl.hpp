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
#include <optional>

#include "oneapi/dal/detail/memory.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/serialization.hpp"

namespace oneapi::dal::detail {
namespace v2 {

/// Serialize the provided array data to the output archive.
/// If the data is on device, the function copies it to host.
///
/// @param[in] policy        The data parallel or default host policy
/// @param[in] archive       The output archive
/// @param[in] data          The pointer to the array data that is written to the
///                          archive. Can be either pure host or USM pointer.
/// @param[in] size_in_bytes The size of the array data buffer in bytes.
/// @param[in] dtype         The type of the values stored in the data buffer
template <typename Policy>
void serialize_array(const Policy& policy,
                     output_archive& archive,
                     const byte_t* data,
                     std::int64_t size_in_bytes,
                     data_type dtype);

/// Deserialize the array data from the input archive.
/// If data parallel policy is provided, the resulting buffer will be allocated on device.
///
/// @param[in] policy         The data parallel or default host policy
/// @param[in] archive        The input archive
/// @param[in] expected_dtype The data type that is expected to be read from archive.
///                           If the expected type does not match the actual data type
///                           stored in the archive, the function throws an exception.
///
/// @return The tuple of shared pointer to the deserialized buffer and its size in bytes.
template <typename Policy>
std::tuple<shared<byte_t>, std::int64_t> deserialize_array(const Policy& policy,
                                                           input_archive& archive,
                                                           data_type expected_dtype);

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
        return new array_impl<T>(policy, data, count, [alloc, count](T* ptr) {
            alloc.deallocate(ptr, count);
        });
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
        if (ref.has_mutable_data()) {
            data_owned_ = shared(ref.get_shared(), data);
        }
        else {
            data_owned_ = shared(ref.get_cshared(), data);
        }
        count_ = count;
        reset_policy(ref);
    }

    template <typename Y>
    void reset(const array_impl<Y>& ref, const T* data, std::int64_t count) {
        if (ref.has_mutable_data()) {
            data_owned_ = cshared(ref.get_shared(), data);
        }
        else {
            data_owned_ = cshared(ref.get_cshared(), data);
        }
        count_ = count;
        reset_policy(ref);
    }

    void serialize(output_archive& ar) const {
        static_assert(is_trivially_serializable_v<T>,
                      "Serialization for non-trivial types is not implemented");

        using data_t = detail::trivial_serialization_type_t<T>;
        const byte_t* data_bytes = reinterpret_cast<const byte_t*>(get_data());
        const data_type dtype = make_data_type<data_t>();
        const std::int64_t size_in_bytes = check_mul_overflow<std::int64_t>(sizeof(data_t), count_);

        __ONEDAL_IF_QUEUE__(get_queue(), {
            ONEDAL_ASSERT(dp_policy_.has_value());
            serialize_array(*dp_policy_, ar, data_bytes, size_in_bytes, dtype);
        });

        __ONEDAL_IF_NO_QUEUE__(get_queue(), {
            serialize_array(default_host_policy{}, ar, data_bytes, size_in_bytes, dtype);
        });
    }

    void deserialize(input_archive& ar) {
        static_assert(is_trivially_serializable_v<T>,
                      "Serialization for non-trivial types is not implemented");

        using data_t = detail::trivial_serialization_type_t<T>;
        const data_type expected_dtype = make_data_type<data_t>();

        detail::shared<byte_t> data_shared;
        std::int64_t size_in_bytes;

        __ONEDAL_IF_QUEUE__(get_queue(), {
            ONEDAL_ASSERT(dp_policy_.has_value());
            std::tie(data_shared, size_in_bytes) =
                deserialize_array(*dp_policy_, ar, expected_dtype);
        });

        __ONEDAL_IF_NO_QUEUE__(get_queue(), {
            std::tie(data_shared, size_in_bytes) =
                deserialize_array(default_host_policy{}, ar, expected_dtype);
        });

        // TODO: Use exception
        ONEDAL_ASSERT(size_in_bytes % sizeof(data_t) == 0);

        data_owned_ = shared{ data_shared, reinterpret_cast<T*>(data_shared.get()) };
        count_ = size_in_bytes / sizeof(data_t);
    }

    shared get_shared() const {
        ONEDAL_ASSERT(!data_owned_.valueless_by_exception());
        return std::get<shared>(data_owned_);
    }

    cshared get_cshared() const {
        ONEDAL_ASSERT(!data_owned_.valueless_by_exception());
        return std::get<cshared>(data_owned_);
    }

#ifdef ONEDAL_DATA_PARALLEL
    std::optional<sycl::queue> get_queue() const {
        if (dp_policy_) {
            return dp_policy_->get_queue();
        }
        return std::nullopt;
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

    shared copy() {
#ifdef ONEDAL_DATA_PARALLEL
        if (dp_policy_.has_value()) {
            const auto policy = dp_policy_.value();
            const auto context = policy.get_queue().get_context();
            const auto alloc_kind = sycl::get_pointer_type(get_data(), context);
            const auto allocator = data_parallel_allocator<T>{ policy, alloc_kind };
            return copy_generic(policy, allocator);
        }
#endif
        return copy_generic(default_host_policy{}, host_allocator<T>{});
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

    const shared* get_if_shared() const noexcept {
        return std::get_if<shared>(&data_owned_);
    }

    std::variant<cshared, shared> data_owned_;
    std::int64_t count_;
#ifdef ONEDAL_DATA_PARALLEL
    std::optional<data_parallel_policy> dp_policy_;
#endif
};

} // namespace v2

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

    array_impl(const shared& data, std::int64_t count) {
        reset(data, count);
    }

    array_impl(const cshared& data, std::int64_t count) {
        reset(data, count);
    }

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

    void reset(const shared& data, std::int64_t count) {
        data_owned_ = data;
        count_ = count;
    }

    void reset(const cshared& data, std::int64_t count) {
        data_owned_ = data;
        count_ = count;
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

    shared get_shared() const {
        ONEDAL_ASSERT(!data_owned_.valueless_by_exception());
        return std::get<shared>(data_owned_);
    }

    cshared get_cshared() const {
        ONEDAL_ASSERT(!data_owned_.valueless_by_exception());
        return std::get<cshared>(data_owned_);
    }

private:
    std::variant<cshared, shared> data_owned_;
    std::int64_t count_;
};

} // namespace v1

using v2::array_impl;

} // namespace oneapi::dal::detail
