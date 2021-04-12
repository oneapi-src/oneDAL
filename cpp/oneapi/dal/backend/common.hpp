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

/// Finds the largest power of 2 number not larger than `x`.
/// Return `x`, if `x` is already power of 2
/// Example: down_pow2(10) == 8
/// Example: down_pow2(16) == 16
template <typename Integer>
inline constexpr Integer down_pow2(Integer x) {
    static_assert(std::is_integral_v<Integer>);
    ONEDAL_ASSERT(x > 0);
    Integer power = 1;
    while (power < x / 2) {
        power *= 2;
    }
    return power;
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
inline bool is_same_context_impl(const sycl::queue& reference, QueueLike&& queue_like) {
    using queue_like_t = std::decay_t<QueueLike>;
    constexpr bool is_queue = std::is_same_v<queue_like_t, sycl::queue>;
    constexpr bool is_opt_queue = std::is_same_v<queue_like_t, std::optional<sycl::queue>>;
    constexpr bool has_get_queue = has_get_queue_v<queue_like_t>;
    constexpr bool has_get_optional_queue = has_get_optional_queue_v<queue_like_t>;

    static_assert(is_queue || is_opt_queue || has_get_queue || has_get_optional_queue,
                  "Unknown object type, cannot extract queue");

    if constexpr (is_queue) {
        return reference.get_context() == queue_like.get_context();
    }
    else if constexpr (is_opt_queue) {
        if (queue_like) {
            return reference.get_context() == queue_like->get_context();
        }
    }
    else if constexpr (has_get_queue) {
        return reference.get_context() == queue_like.get_queue().get_context();
    }
    else if constexpr (has_get_optional_queue) {
        auto optional_queue = queue_like.get_queue();
        if (optional_queue) {
            return reference.get_context() == optional_queue->get_context();
        }
    }

    return true;
}

template <typename QueueLike>
inline bool is_same_device_impl(const sycl::queue& reference, QueueLike&& queue_like) {
    using queue_like_t = std::decay_t<QueueLike>;
    constexpr bool is_queue = std::is_same_v<queue_like_t, sycl::queue>;
    constexpr bool is_opt_queue = std::is_same_v<queue_like_t, std::optional<sycl::queue>>;
    constexpr bool has_get_queue = has_get_queue_v<queue_like_t>;
    constexpr bool has_get_optional_queue = has_get_optional_queue_v<queue_like_t>;

    static_assert(is_queue || is_opt_queue || has_get_queue || has_get_optional_queue,
                  "Unknown object type, cannot extract queue");

    if constexpr (is_queue) {
        return reference.get_device() == queue_like.get_device();
    }
    else if constexpr (is_opt_queue) {
        if (queue_like) {
            return reference.get_device() == queue_like->get_device();
        }
    }
    else if constexpr (has_get_queue) {
        return reference.get_device() == queue_like.get_queue().get_device();
    }
    else if constexpr (has_get_optional_queue) {
        auto optional_queue = queue_like.get_queue();
        if (optional_queue) {
            return reference.get_device() == optional_queue->get_device();
        }
    }

    ONEDAL_ASSERT(!"Unreachable");
    return false;
}

template <typename... QueueLike>
inline bool is_same_context(const sycl::queue& reference, QueueLike&&... queues_like) {
    return (... && is_same_context_impl(reference, std::forward<QueueLike>(queues_like)));
}

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
