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

template <typename Type, typename Index>
sycl::event argmin(sycl::queue& queue, 
                   const ndview<Type, 1>& values, 
                   ndview<Type, 1>& val_output, 
                   ndview<Index, 1>& idx_output, 
                   const event_vector& deps) {
    ONEDAL_ASSERT(values.has_data());
    ONEDAL_ASSERT(val_output.has_mutable_data());
    ONEDAL_ASSERT(idx_output.has_mutable_data());
    ONEDAL_ASSERT(std::int64_t(1) == val_output.get_count());
    ONEDAL_ASSERT(std::int64_t(1) == idx_output.get_count());

    constexpr min<Type> binary{};
    constexpr identity<Type> unary{};
    
    const auto* const ptr = val_output.get_mutable_data();
    auto min_event = reduce_1d(queue, values, val_output, binary, unary, deps);
    auto functor = [=](Type candidate) -> bool { return (*ptr) == candidate; };
    return argwhere_one(queue, functor, values, idx_output, { min_event });
}

template <typename Type, typename Index>
std::tuple<Type, Index> argmin(sycl::queue& queue, 
                               const ndview<Type, 1>& values, 
                               const event_vector& deps) {
    using dal::backend::operator+;
    constexpr auto alloc = sycl::usm::alloc::device;
    auto [val_output, val_event] = ndarray<Type, 1>::zeros(queue, { 1 }, alloc);
    auto [idx_output, idx_event] = ndarray<Index, 1>::zeros(queue, { 1 }, alloc);
    auto event = argmin(queue, values, val_output, idx_output, deps + val_event + idx_event);
    return { val_output.at_device(queue, 0, { event }), idx_output.at_device(queue, 0, { event }) };
}

#define INSTANTIATE(T, I)                                   \
    template sycl::event argmin(sycl::queue&, const ndview<T, 1>&, ndview<T, 1>&, ndview<I, 1>&, const event_vector&); \
    template std::tuple<T, I> argmin(sycl::queue&, const ndview<T, 1>&, const event_vector&);


INSTANTIATE(float, std::int64_t)
INSTANTIATE(double, std::int64_t)

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
