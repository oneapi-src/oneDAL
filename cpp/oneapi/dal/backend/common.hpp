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

#ifdef ONEDAL_DATA_PARALLEL
#include <CL/sycl.hpp>
#endif

#include "oneapi/dal/array.hpp"
#include "oneapi/dal/detail/common.hpp"

#if defined(__INTEL_COMPILER)
#define PRAGMA_IVDEP         _Pragma("ivdep")
#define PRAGMA_VECTOR_ALWAYS _Pragma("vector always")
#else
#define PRAGMA_IVDEP
#define PRAGMA_VECTOR_ALWAYS
#endif

namespace oneapi::dal::backend {

template <std::int64_t axis_count>
using ndindex = std::array<std::int64_t, axis_count>;

/// Finds the largest multiple of `multiple` not larger than `x`
/// Return `x`, if `x` is already multiple of `multiple`
/// Example: down_multiple(10, 4) == 8
/// Example: down_multiple(10, 5) == 10
template <typename Integer>
inline constexpr Integer down_multiple(Integer x, Integer multiple) {
    static_assert(std::is_integral_v<Integer>);
    ONEDAL_ASSERT(x > 0);
    ONEDAL_ASSERT(multiple > 0);
    return (x / multiple) * multiple;
}

/// Finds the smallest multiple of `multiple` not smaller than `x`.
/// Return `x`, if `x` is already multiple of `multiple`
/// Example: up_multiple(10, 4) == 12
/// Example: up_multiple(10, 5) == 10
template <typename Integer>
inline constexpr Integer up_multiple(Integer x, Integer multiple) {
    static_assert(std::is_integral_v<Integer>);
    ONEDAL_ASSERT(x > 0);
    ONEDAL_ASSERT(multiple > 0);
    const Integer y = down_multiple<Integer>(x, multiple);
    const Integer z = multiple * Integer((x % multiple) != 0);
    ONEDAL_ASSERT_SUM_OVERFLOW(Integer, y, z);
    return y + z;
}

/// Checks if 'x' is power of 2.
/// Example: down_pow2(10) == false
/// Example: down_pow2(16) == true
template <typename Integer>
inline constexpr bool is_pow2(Integer x) {
    static_assert(std::is_integral_v<Integer>);
    return !(x & (x - 1)) && (x > 0) ? true : false;
}

template <typename Integer>
inline constexpr std::int64_t get_magnitude_bit_count() {
    static_assert(std::is_integral_v<Integer>);
    constexpr std::int64_t bit_count = sizeof(Integer) * 8;
    return std::is_signed_v<Integer> ? bit_count - 1 : bit_count;
}

/// Finds the largest power of 2 number not larger than `x`.
/// Return `x`, if `x` is already power of 2
/// Example: down_pow2(10) == 8
/// Example: down_pow2(16) == 16
template <typename Integer>
inline constexpr Integer down_pow2(Integer x) {
    static_assert(std::is_integral_v<Integer>);
    ONEDAL_ASSERT(x > 0);
    Integer power = 0;
    if (is_pow2(x)) {
        return x;
    }

    while (x > 1) {
        x >>= 1;
        power++;
    }
    ONEDAL_ASSERT(power < get_magnitude_bit_count<Integer>());
    return 1 << power;
}

/// Finds the smallest power of 2 number not smaller than `x`.
/// Return `x`, if `x` is already power of 2
/// Example: up_pow2(10) == 16
/// Example: up_pow2(16) == 16
template <typename Integer>
inline constexpr Integer up_pow2(Integer x) {
    static_assert(std::is_integral_v<Integer>);
    ONEDAL_ASSERT(x > 0);
    Integer power = 1;
    while (power < x) {
        ONEDAL_ASSERT_MUL_OVERFLOW(Integer, power, 2);
        power *= 2;
    }
    return power;
}

#ifdef ONEDAL_DATA_PARALLEL

using event_vector = std::vector<sycl::event>;

template <typename T>
class object_store_entry : public base {
public:
    object_store_entry(const T& value) : inner_(value) {}
    object_store_entry(T&& value) : inner_(std::move(value)) {}

private:
    T inner_;
};

class object_store : public base {
public:
    using container_t = std::vector<base*>;

    ~object_store() {
        for (auto obj : container_) {
            delete obj;
        }
    }

    template <typename T>
    void add(T&& obj) {
        container_.push_back(new object_store_entry{ std::forward<T>(obj) });
    }

    void clear() {
        container_.clear();
    }

private:
    container_t container_;
};

class smart_event : public base {
public:
    smart_event() = default;
    smart_event(const sycl::event& event) : event_(event) {}
    smart_event(sycl::event&& event) : event_(std::move(event)) {}

    operator sycl::event() const {
        return event_;
    }

    smart_event& operator=(const sycl::event& event) {
        event_ = event;
        return *this;
    }

    smart_event& operator=(sycl::event&& event) {
        event_ = std::move(event);
        return *this;
    }

    void wait() {
        event_.wait();
        store_->clear();
    }

    void wait_and_throw() {
        event_.wait_and_throw();
        store_->clear();
    }

    template <typename T>
    smart_event& attach(T&& value) {
        store_->add(std::forward<T>(value));
        return *this;
    }

private:
    sycl::event event_;
    std::shared_ptr<object_store> store_ = std::make_shared<object_store>();
};

inline bool is_same_context(const sycl::queue& q1, const sycl::queue& q2) {
    return q1.get_context() == q2.get_context();
}

template <typename, typename = void>
struct has_get_queue : std::false_type {};

template <typename T>
struct has_get_queue<T, std::void_t<decltype(&T::get_queue)>>
        : std::is_same<sycl::queue, std::decay_t<decltype(std::declval<T>().get_queue())>> {};

template <typename, typename = void>
struct has_get_optional_queue : std::false_type {};

template <typename T>
struct has_get_optional_queue<T, std::void_t<decltype(&T::get_queue)>>
        : std::is_same<std::optional<sycl::queue>,
                       std::decay_t<decltype(std::declval<T>().get_queue())>> {};

template <typename T>
inline constexpr bool has_get_queue_v = has_get_queue<T>::value;

template <typename T>
inline constexpr bool has_get_optional_queue_v = has_get_optional_queue<T>::value;

template <typename QueueLike>
struct queue_like_traits {
    using queue_like_t = std::decay_t<QueueLike>;
    static constexpr bool is_queue = std::is_same_v<queue_like_t, sycl::queue>;
    static constexpr bool is_optional_queue =
        std::is_same_v<queue_like_t, std::optional<sycl::queue>>;
    static constexpr bool has_get_queue = has_get_queue_v<queue_like_t>;
    static constexpr bool has_get_optional_queue = has_get_optional_queue_v<queue_like_t>;
    static constexpr bool is_valid =
        is_queue || is_optional_queue || has_get_queue || has_get_optional_queue;
};

template <typename QueueLike>
inline std::optional<sycl::queue> extract_queue(QueueLike&& queue_like) {
    static_assert(queue_like_traits<QueueLike>::is_valid,
                  "Invalid queue-like object, cannot extract queue");

    if constexpr (queue_like_traits<QueueLike>::is_queue ||
                  queue_like_traits<QueueLike>::is_optional_queue) {
        return queue_like;
    }
    else if constexpr (queue_like_traits<QueueLike>::has_get_queue ||
                       queue_like_traits<QueueLike>::has_get_optional_queue) {
        return queue_like.get_queue();
    }

    return std::nullopt;
}

template <typename QueueLike>
inline bool is_same_context_impl(const sycl::queue& reference, QueueLike&& queue_like) {
    const auto optional_queue = extract_queue(std::forward<QueueLike>(queue_like));
    return optional_queue && (optional_queue->get_context() == reference.get_context());
}

template <typename QueueLike>
inline bool is_same_context_ignore_nullopt_impl(const sycl::queue& reference,
                                                QueueLike&& queue_like) {
    const auto optional_queue = extract_queue(std::forward<QueueLike>(queue_like));
    if (optional_queue) {
        return optional_queue->get_context() == reference.get_context();
    }
    return true;
}

template <typename QueueLike>
inline bool is_same_device_impl(const sycl::queue& reference, QueueLike&& queue_like) {
    const auto optional_queue = extract_queue(std::forward<QueueLike>(queue_like));
    return optional_queue && (optional_queue->get_device() == reference.get_device());
}

/// Checks whether all queue-like objects have the same context.
template <typename... QueueLike>
inline bool is_same_context(const sycl::queue& reference, QueueLike&&... queues_like) {
    return (... && is_same_context_impl(reference, std::forward<QueueLike>(queues_like)));
}

/// Checks whether all queue-like objects have the same context. The queue-like
/// objects, which does not carry context, do not participate in comparison.
template <typename... QueueLike>
inline bool is_same_context_ignore_nullopt(const sycl::queue& reference,
                                           QueueLike&&... queues_like) {
    return (... &&
            is_same_context_ignore_nullopt_impl(reference, std::forward<QueueLike>(queues_like)));
}

/// Checks whether all queue-like objects have the same device.
template <typename... QueueLike>
inline bool is_same_device(const sycl::queue& reference, QueueLike&&... queues_like) {
    return (... && is_same_device_impl(reference, std::forward<QueueLike>(queues_like)));
}

template <typename... QueueLike>
inline void check_if_same_context(const sycl::queue& reference, QueueLike&&... queues_like) {
    if (!is_same_context(reference, std::forward<QueueLike>(queues_like)...)) {
        throw invalid_argument{ dal::detail::error_messages::queues_in_different_contexts() };
    }
}

inline sycl::range<1> make_range_1d(std::int64_t size) {
    return { dal::detail::integral_cast<std::size_t>(size) };
}

inline sycl::range<2> make_range_2d(std::int64_t size1, std::int64_t size2) {
    return { dal::detail::integral_cast<std::size_t>(size1),
             dal::detail::integral_cast<std::size_t>(size2) };
}

/// Creates `nd_range`, where global size is multiple of local size
inline sycl::nd_range<1> make_multiple_nd_range_1d(std::int64_t global_size,
                                                   std::int64_t local_size) {
    const auto g = dal::detail::integral_cast<std::size_t>(global_size);
    const auto l = dal::detail::integral_cast<std::size_t>(local_size);
    return { up_multiple(g, l), l };
}

/// Creates `nd_range`, where global sizes is multiple of local size
inline sycl::nd_range<2> make_multiple_nd_range_2d(const ndindex<2>& global_size,
                                                   const ndindex<2>& local_size) {
    const auto g_0 = dal::detail::integral_cast<std::size_t>(global_size[0]);
    const auto g_1 = dal::detail::integral_cast<std::size_t>(global_size[1]);
    const auto l_0 = dal::detail::integral_cast<std::size_t>(local_size[0]);
    const auto l_1 = dal::detail::integral_cast<std::size_t>(local_size[1]);
    return { { up_multiple(g_0, l_0), up_multiple(g_1, l_1) }, { l_0, l_1 } };
}

/// Creates `nd_range`, where global sizes is multiple of local size
inline sycl::nd_range<3> make_multiple_nd_range_3d(const ndindex<3>& global_size,
                                                   const ndindex<3>& local_size) {
    const auto g_0 = dal::detail::integral_cast<std::size_t>(global_size[0]);
    const auto g_1 = dal::detail::integral_cast<std::size_t>(global_size[1]);
    const auto g_2 = dal::detail::integral_cast<std::size_t>(global_size[2]);
    const auto l_0 = dal::detail::integral_cast<std::size_t>(local_size[0]);
    const auto l_1 = dal::detail::integral_cast<std::size_t>(local_size[1]);
    const auto l_2 = dal::detail::integral_cast<std::size_t>(local_size[2]);
    return { { up_multiple(g_0, l_0), up_multiple(g_1, l_1), up_multiple(g_2, l_2) },
             { l_0, l_1, l_2 } };
}

inline std::int64_t device_max_wg_size(const sycl::queue& q) {
    const auto res = q.get_device().template get_info<sycl::info::device::max_work_group_size>();
    return dal::detail::integral_cast<std::int64_t>(res);
}

inline std::int64_t device_max_sg_size(const sycl::queue& q) {
    const std::vector<size_t> sg_sizes =
        q.get_device().template get_info<sycl::info::device::sub_group_sizes>();
    auto result_iter = std::max_element(sg_sizes.begin(), sg_sizes.end());
    ONEDAL_ASSERT(result_iter != sg_sizes.end());
    return dal::detail::integral_cast<std::int64_t>(*result_iter);
}

inline std::int64_t propose_wg_size(const sycl::queue& q) {
    // TODO: a temporary solution that limits work item count used on the device.
    // Needs to change to more smart logic in the future.
    return std::min<std::int64_t>(512, device_max_wg_size(q));
}

inline std::int64_t device_local_mem_size(const sycl::queue& q) {
    const auto res = q.get_device().template get_info<sycl::info::device::local_mem_size>();
    return dal::detail::integral_cast<std::int64_t>(res);
}

template <typename T>
inline std::int64_t device_native_vector_size(const sycl::queue& q) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
    return 0;
}

template <>
inline std::int64_t device_native_vector_size<float>(const sycl::queue& q) {
    const auto res =
        q.get_device().template get_info<sycl::info::device::native_vector_width_float>();
    return dal::detail::integral_cast<std::int64_t>(res);
}

template <>
inline std::int64_t device_native_vector_size<double>(const sycl::queue& q) {
    const auto res =
        q.get_device().template get_info<sycl::info::device::native_vector_width_double>();
    return dal::detail::integral_cast<std::int64_t>(res);
}

#endif

} // namespace oneapi::dal::backend
