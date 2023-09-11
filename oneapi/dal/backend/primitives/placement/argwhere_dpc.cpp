/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"

#include "oneapi/dal/backend/primitives/placement/argwhere.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Binary, typename Type, typename Index, where_alignment align>
sycl::event argextreme(sycl::queue& queue,
                       const Binary& binary,
                       const ndview<Type, 1>& values,
                       ndview<Type, 1>& val_output,
                       ndview<Index, 1>& idx_output,
                       const event_vector& deps) {
    ONEDAL_ASSERT(values.has_data());
    ONEDAL_ASSERT(val_output.has_mutable_data());
    ONEDAL_ASSERT(idx_output.has_mutable_data());
    ONEDAL_ASSERT(std::int64_t(1) == val_output.get_count());
    ONEDAL_ASSERT(std::int64_t(1) == idx_output.get_count());

    constexpr identity<Type> unary{};

    const auto* const ptr = val_output.get_mutable_data();
    auto min_event = reduce_1d(queue, values, val_output, binary, unary, deps);
    auto functor = [ptr](Type candidate) -> bool {
        return *ptr == candidate;
    };
    return argwhere_one(queue, functor, values, idx_output, align, { min_event });
}

template <typename Binary, typename Type, typename Index, where_alignment align>
std::tuple<Type, Index> argextreme(sycl::queue& queue,
                                   const Binary& binary,
                                   const ndview<Type, 1>& values,
                                   const event_vector& deps) {
    using dal::backend::operator+;

    constexpr auto alloc = sycl::usm::alloc::device;
    constexpr auto val_identity = Binary::init_value;
    constexpr auto idx_identity = where_alignment_map<Index, align>::identity;

    auto [idx_output, idx_event] = ndarray<Index, 1>::full(queue, { 1 }, { idx_identity }, alloc);
    auto [val_output, val_event] = ndarray<Type, 1>::full(queue, { 1 }, { val_identity }, alloc);

    auto event = argextreme<Binary, Type, Index, align>(queue,
                                                        binary,
                                                        values, //
                                                        val_output,
                                                        idx_output,
                                                        deps + val_event + idx_event);
    return { val_output.at_device(queue, 0, { event }), idx_output.at_device(queue, 0, { event }) };
}

template <typename Type, typename Index, where_alignment align>
sycl::event argmin(sycl::queue& queue,
                   const ndview<Type, 1>& values,
                   ndview<Type, 1>& val_output,
                   ndview<Index, 1>& idx_output,
                   const event_vector& deps) {
    constexpr min<Type> binary{};
    return argextreme<decltype(binary), Type, Index, align>(queue,
                                                            binary,
                                                            values,
                                                            val_output,
                                                            idx_output,
                                                            deps);
}

template <typename Type, typename Index>
sycl::event argmin(sycl::queue& queue,
                   const ndview<Type, 1>& values,
                   ndview<Type, 1>& val_output,
                   ndview<Index, 1>& idx_output,
                   where_alignment align,
                   const event_vector& deps) {
    constexpr auto left = where_alignment::left;
    constexpr auto right = where_alignment::right;

    if (align == right) {
        return argmin<Type, Index, right>(queue, values, val_output, idx_output, deps);
    }
    if (align == left) {
        return argmin<Type, Index, left>(queue, values, val_output, idx_output, deps);
    }
    ONEDAL_ASSERT(false);
    return sycl::event{};
}

template <typename Type, typename Index, where_alignment align>
std::tuple<Type, Index> argmin(sycl::queue& queue,
                               const ndview<Type, 1>& values,
                               const event_vector& deps) {
    constexpr min<Type> binary{};
    return argextreme<decltype(binary), Type, Index, align>(queue, binary, values, deps);
}

template <typename Type, typename Index>
std::tuple<Type, Index> argmin(sycl::queue& queue,
                               const ndview<Type, 1>& values,
                               where_alignment align,
                               const event_vector& deps) {
    constexpr auto left = where_alignment::left;
    constexpr auto right = where_alignment::right;

    if (align == right) {
        return argmin<Type, Index, right>(queue, values, deps);
    }
    if (align == left) {
        return argmin<Type, Index, left>(queue, values, deps);
    }
    ONEDAL_ASSERT(false);
    return std::tuple<Type, Index>(0, -1);
}

template <typename Type, typename Index, where_alignment align>
sycl::event argmax(sycl::queue& queue,
                   const ndview<Type, 1>& values,
                   ndview<Type, 1>& val_output,
                   ndview<Index, 1>& idx_output,
                   const event_vector& deps) {
    constexpr max<Type> binary{};
    return argextreme<decltype(binary), Type, Index, align>(queue,
                                                            binary,
                                                            values,
                                                            val_output,
                                                            idx_output,
                                                            deps);
}

template <typename Type, typename Index>
sycl::event argmax(sycl::queue& queue,
                   const ndview<Type, 1>& values,
                   ndview<Type, 1>& val_output,
                   ndview<Index, 1>& idx_output,
                   where_alignment align,
                   const event_vector& deps) {
    constexpr auto left = where_alignment::left;
    constexpr auto right = where_alignment::right;

    if (align == right) {
        return argmax<Type, Index, right>(queue, values, val_output, idx_output, deps);
    }
    if (align == left) {
        return argmax<Type, Index, left>(queue, values, val_output, idx_output, deps);
    }
    ONEDAL_ASSERT(false);
    return sycl::event{};
}

template <typename Type, typename Index, where_alignment align>
std::tuple<Type, Index> argmax(sycl::queue& queue,
                               const ndview<Type, 1>& values,
                               const event_vector& deps) {
    constexpr max<Type> binary{};
    return argextreme<decltype(binary), Type, Index, align>(queue, binary, values, deps);
}

template <typename Type, typename Index>
std::tuple<Type, Index> argmax(sycl::queue& queue,
                               const ndview<Type, 1>& values,
                               where_alignment align,
                               const event_vector& deps) {
    constexpr auto left = where_alignment::left;
    constexpr auto right = where_alignment::right;

    if (align == right) {
        return argmax<Type, Index, right>(queue, values, deps);
    }
    if (align == left) {
        return argmax<Type, Index, left>(queue, values, deps);
    }
    ONEDAL_ASSERT(false);
    return std::tuple<Type, Index>(0, -1);
}

#define INSTANTIATE(T, I, A)                                        \
    template sycl::event argmin<T, I, A>(sycl::queue&,              \
                                         const ndview<T, 1>&,       \
                                         ndview<T, 1>&,             \
                                         ndview<I, 1>&,             \
                                         const event_vector&);      \
    template sycl::event argmax<T, I, A>(sycl::queue&,              \
                                         const ndview<T, 1>&,       \
                                         ndview<T, 1>&,             \
                                         ndview<I, 1>&,             \
                                         const event_vector&);      \
    template std::tuple<T, I> argmin<T, I, A>(sycl::queue&,         \
                                              const ndview<T, 1>&,  \
                                              const event_vector&); \
    template std::tuple<T, I> argmax<T, I, A>(sycl::queue&,         \
                                              const ndview<T, 1>&,  \
                                              const event_vector&);

#define INSTANTIATE_ALIGN(T, I)                            \
    INSTANTIATE(T, I, where_alignment::right)              \
    INSTANTIATE(T, I, where_alignment::left)               \
    template sycl::event argmin(sycl::queue&,              \
                                const ndview<T, 1>&,       \
                                ndview<T, 1>&,             \
                                ndview<I, 1>&,             \
                                where_alignment,           \
                                const event_vector&);      \
    template sycl::event argmax(sycl::queue&,              \
                                const ndview<T, 1>&,       \
                                ndview<T, 1>&,             \
                                ndview<I, 1>&,             \
                                where_alignment,           \
                                const event_vector&);      \
    template std::tuple<T, I> argmin(sycl::queue&,         \
                                     const ndview<T, 1>&,  \
                                     where_alignment,      \
                                     const event_vector&); \
    template std::tuple<T, I> argmax(sycl::queue&,         \
                                     const ndview<T, 1>&,  \
                                     where_alignment,      \
                                     const event_vector&);

INSTANTIATE_ALIGN(float, std::int32_t)
INSTANTIATE_ALIGN(double, std::int32_t)
INSTANTIATE_ALIGN(float, std::int64_t)
INSTANTIATE_ALIGN(double, std::int64_t)

} // namespace oneapi::dal::backend::primitives
