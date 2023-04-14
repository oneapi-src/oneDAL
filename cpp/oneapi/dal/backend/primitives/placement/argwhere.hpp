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

#ifdef ONEDAL_DATA_PARALLEL

template <typename Functor, typename Type, typename Index>
inline sycl::event argwhere_one(sycl::queue& queue,
                                const Functor& unary,
                                const ndview<Type, 1>& values,
                                ndview<Index, 1>& output,
                                const event_vector& deps = {}) {
    ONEDAL_ASSERT(values.has_data());
    ONEDAL_ASSERT(output.has_mutable_data());
    ONEDAL_ASSERT(std::int64_t(1) == output.get_count());

    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);

        const auto count = values.get_count();
        const auto range = make_range_1d(count);
        const auto* const inp_ptr = values.get_data();
        auto* const out_ptr = output.get_mutable_data();
        
        constexpr sycl::ext::oneapi::maximum<Index> max{};
        constexpr auto identity = std::numeric_limits<Index>::lowest();
        
        auto accum = sycl::reduction(out_ptr, identity, max);

        h.parallel_for(range, accum, [=](sycl::id<1> idx, auto& acc) {
            const auto value = inp_ptr[idx];
            const bool handle = unary(value);
            const std::int64_t arg = handle 
                        ? std::int64_t(idx)
                        : std::int64_t(-1l);  
            acc.combine(arg);
        });
    });
}

template <typename Functor, typename Type, typename Index = std::int64_t>
inline Index argwhere_one(sycl::queue& queue,
                          const Functor& unary,
                          const ndview<Type, 1>& values,
                          const event_vector& deps = {}) {
    using dal::backend::operator+;
    constexpr auto alloc = sycl::usm::alloc::device;
    constexpr auto identity = std::numeric_limits<Index>::lowest();
    auto [out, out_event] = ndarray<Index, 1>::full(queue, { 1 }, identity, alloc);
    auto event = argwhere_one(queue, unary, values, out, deps + out_event);
    return out.at_device(queue, 0, { event });
}

template <typename Type, typename Index>
sycl::event argmin(sycl::queue& queue, 
                   const ndview<Type, 1>& values, 
                   ndview<Type, 1>& val_output, 
                   ndview<Index, 1>& idx_output, 
                   const event_vector& deps = {});

template <typename Type, typename Index = std::int64_t>
std::tuple<Type, Index> argmin(sycl::queue& queue, 
                               const ndview<Type, 1>& values, 
                               const event_vector& deps = {});

#endif

} // namespace oneapi::dal::backend::primitives
