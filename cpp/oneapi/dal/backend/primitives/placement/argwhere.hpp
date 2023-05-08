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

#pragma once

#include <tuple>
#include <limits>

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"

namespace oneapi::dal::backend::primitives {

enum class where_alignment : std::int64_t { left = 0b0l, right = 0b1l };

#ifdef ONEDAL_DATA_PARALLEL

template <typename Index, where_alignment align>
struct where_alignment_map {};

template <typename Index>
struct where_alignment_map<Index, where_alignment::left> {
    constexpr static inline sycl::ext::oneapi::maximum<Index> binary{};
    constexpr static inline auto identity = std::numeric_limits<Index>::lowest();
};

template <typename Index>
struct where_alignment_map<Index, where_alignment::right> {
    constexpr static inline sycl::ext::oneapi::minimum<Index> binary{};
    constexpr static inline auto identity = std::numeric_limits<Index>::max();
};

template <typename Functor,
          typename Type,
          typename Index,
          where_alignment align = where_alignment::right>
inline sycl::event argwhere_one(sycl::queue& queue,
                                const Functor& unary,
                                const ndview<Type, 1>& values,
                                ndview<Index, 1>& output,
                                const event_vector& deps = {}) {
    ONEDAL_ASSERT(values.has_data());
    ONEDAL_ASSERT(output.has_mutable_data());
    ONEDAL_ASSERT(std::int64_t(1) == output.get_count());

    using map_t = where_alignment_map<Index, align>;

    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);

        const auto count = values.get_count();
        const auto range = make_range_1d(count);
        const auto* const inp_ptr = values.get_data();
        auto* const out_ptr = output.get_mutable_data();

        constexpr auto binary = map_t::binary;
        constexpr auto identity = map_t::identity;

        auto accum = sycl::reduction(out_ptr, identity, binary);

        h.parallel_for(range, accum, [=](sycl::id<1> idx, auto& acc) {
            const bool handle = unary(inp_ptr[idx]);

            if (handle)
                acc.combine(Index(idx));
        });
    });
}

template <typename Functor, typename Type, typename Index>
inline sycl::event argwhere_one(sycl::queue& queue,
                                const Functor& unary,
                                const ndview<Type, 1>& values,
                                ndview<Index, 1>& output,
                                where_alignment align,
                                const event_vector& deps = {}) {
    constexpr auto right = where_alignment::right;
    constexpr auto left = where_alignment::left;

    if (align == right) {
        return argwhere_one<Functor, Type, Index, right>( //
            queue,
            unary,
            values,
            output,
            deps);
    }
    if (align == left) {
        return argwhere_one<Functor, Type, Index, left>( //
            queue,
            unary,
            values,
            output,
            deps);
    }
    ONEDAL_ASSERT(false);
    return sycl::event{};
}

template <typename Functor,
          typename Type,
          typename Index = std::int64_t,
          where_alignment align = where_alignment::right>
inline Index argwhere_one(sycl::queue& queue,
                          const Functor& unary,
                          const ndview<Type, 1>& values,
                          const event_vector& deps = {}) {
    using dal::backend::operator+;
    constexpr auto alloc = sycl::usm::alloc::device;
    constexpr auto identity = where_alignment_map<Index, align>::identity;
    auto [out, out_event] = ndarray<Index, 1>::full(queue, { 1 }, identity, alloc);
    auto event = argwhere_one<Functor, Type, Index, align>( //
        queue,
        unary,
        values,
        out,
        deps + out_event);
    return out.at_device(queue, 0, { event });
}

template <typename Functor, typename Type, typename Index = std::int64_t>
inline Index argwhere_one(sycl::queue& queue,
                          const Functor& unary,
                          const ndview<Type, 1>& values,
                          where_alignment align,
                          const event_vector& deps = {}) {
    constexpr auto right = where_alignment::right;
    constexpr auto left = where_alignment::left;

    if (align == right) {
        return argwhere_one<Functor, Type, Index, right>( //
            queue,
            unary,
            values,
            deps);
    }
    if (align == left) {
        return argwhere_one<Functor, Type, Index, left>( //
            queue,
            unary,
            values,
            deps);
    }
    ONEDAL_ASSERT(false);
    return -1l;
}

template <typename Type, typename Index, where_alignment align = where_alignment::right>
sycl::event argmin(sycl::queue& queue,
                   const ndview<Type, 1>& values,
                   ndview<Type, 1>& val_output,
                   ndview<Index, 1>& idx_output,
                   const event_vector& deps = {});

template <typename Type, typename Index, where_alignment align = where_alignment::right>
sycl::event argmax(sycl::queue& queue,
                   const ndview<Type, 1>& values,
                   ndview<Type, 1>& val_output,
                   ndview<Index, 1>& idx_output,
                   const event_vector& deps = {});

template <typename Type, typename Index>
sycl::event argmin(sycl::queue& queue,
                   const ndview<Type, 1>& values,
                   ndview<Type, 1>& val_output,
                   ndview<Index, 1>& idx_output,
                   where_alignment align,
                   const event_vector& deps = {});

template <typename Type, typename Index>
sycl::event argmax(sycl::queue& queue,
                   const ndview<Type, 1>& values,
                   ndview<Type, 1>& val_output,
                   ndview<Index, 1>& idx_output,
                   where_alignment align,
                   const event_vector& deps = {});

template <typename Type,
          typename Index = std::int64_t,
          where_alignment align = where_alignment::right>
std::tuple<Type, Index> argmin(sycl::queue& queue,
                               const ndview<Type, 1>& values,
                               const event_vector& deps = {});

template <typename Type,
          typename Index = std::int64_t,
          where_alignment align = where_alignment::right>
std::tuple<Type, Index> argmax(sycl::queue& queue,
                               const ndview<Type, 1>& values,
                               const event_vector& deps = {});

template <typename Type, typename Index = std::int64_t>
std::tuple<Type, Index> argmin(sycl::queue& queue,
                               const ndview<Type, 1>& values,
                               where_alignment align,
                               const event_vector& deps = {});

template <typename Type, typename Index = std::int64_t>
std::tuple<Type, Index> argmax(sycl::queue& queue,
                               const ndview<Type, 1>& values,
                               where_alignment align,
                               const event_vector& deps = {});

#endif

} // namespace oneapi::dal::backend::primitives
