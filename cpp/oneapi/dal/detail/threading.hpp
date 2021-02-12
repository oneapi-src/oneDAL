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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::preview {
typedef void (*functype)(std::int32_t i, const void *a);

typedef std::int64_t (*loop_functype_size_t_int64)(size_t start_idx,
                                            size_t end_idx,
                                            std::int64_t value_for_reduce,
                                            const void *a);

typedef std::int64_t (*loop_functype_int32ptr_int64)(const int32_t* begin,
                                            const int32_t* end,
                                            std::int64_t value_for_reduce,
                                            const void *a);

typedef std::int64_t (*reduction_functype_int64)(std::int64_t a,
                                                 std::int64_t b,
                                                 const void *reduction);
} // namespace oneapi::dal::preview

extern "C" {
ONEDAL_EXPORT void _onedal_threader_for(std::int32_t n,
                                        std::int32_t threads_request,
                                        const void *a,
                                        oneapi::dal::preview::functype func);

ONEDAL_EXPORT std::int64_t _onedal_parallel_reduce_size_t_int64(
    size_t n,
    std::int64_t init,
    const void *a,
    oneapi::dal::preview::loop_functype_size_t_int64 loop_func,
    const void *b,
    oneapi::dal::preview::reduction_functype_int64 reduction_func);

ONEDAL_EXPORT std::int64_t _onedal_parallel_reduce_size_t_int64_simple(
    size_t n,
    std::int64_t init,
    const void *a,
    oneapi::dal::preview::loop_functype_size_t_int64 loop_func,
    const void *b,
    oneapi::dal::preview::reduction_functype_int64 reduction_func);

ONEDAL_EXPORT std::int64_t _onedal_parallel_reduce_int32ptr_int64_simple(
    const std::int32_t* begin,
    const std::int32_t* end,
    std::int64_t init,
    const void *a,
    oneapi::dal::preview::loop_functype_int32ptr_int64 loop_func,
    const void *b,
    oneapi::dal::preview::reduction_functype_int64 reduction_func);
}

namespace oneapi::dal::detail {
template <typename F>
inline void threader_func(std::int32_t i, const void *a) {
    const F &lambda = *static_cast<const F *>(a);
    lambda(i);
}

template <typename F>
inline ONEDAL_EXPORT void threader_for(std::int32_t n,
                                       std::int32_t threads_request,
                                       const F &lambda) {
    const void *a = static_cast<const void *>(&lambda);

    _onedal_threader_for(n, threads_request, a, threader_func<F>);
}

template <typename F>
inline std::int64_t parallel_reduce_loop_size_t_int64(size_t start_idx,
                                               size_t end_idx,
                                               std::int64_t value_for_reduce,
                                               const void *a) {
    const F &lambda = *static_cast<const F *>(a);
    return lambda(start_idx, end_idx, value_for_reduce);
}

template <typename F>
inline std::int64_t parallel_reduce_loop_int32ptr_int64(const std::int32_t* start_idx,
                                               const std::int32_t* end_idx,
                                               std::int64_t value_for_reduce,
                                               const void *a) {
    const F &lambda = *static_cast<const F *>(a);
    return lambda(start_idx, end_idx, value_for_reduce);
}

template <typename F>
inline std::int64_t parallel_reduce_reduction_int64(std::int64_t a,
                                                    std::int64_t b,
                                                    const void *reduction) {
    const F &lambda = *static_cast<const F *>(reduction);
    return lambda(a, b);
}

template <typename Value, typename Func, typename Reduction>
ONEDAL_EXPORT Value
parallel_reduce_size_t_int64_t(size_t n, Value init, const Func &func, const Reduction &reduction) {
    const void *const lf = static_cast<const void *>(&func);
    const void *const rf = static_cast<const void *>(&reduction);

    return _onedal_parallel_reduce_size_t_int64(n,
                                         init,
                                         lf,
                                         parallel_reduce_loop_size_t_int64<Func>,
                                         rf,
                                         parallel_reduce_reduction_int64<Reduction>);
}

template <typename Value, typename Func, typename Reduction>
ONEDAL_EXPORT Value
parallel_reduce_size_t_int64_t_simple(size_t n, Value init, const Func &func, const Reduction &reduction) {
    const void *const lf = static_cast<const void *>(&func);
    const void *const rf = static_cast<const void *>(&reduction);

    return _onedal_parallel_reduce_size_t_int64_simple(n,
                                         init,
                                         lf,
                                         parallel_reduce_loop_size_t_int64<Func>,
                                         rf,
                                         parallel_reduce_reduction_int64<Reduction>);
}

template <typename Value, typename Func, typename Reduction>
ONEDAL_EXPORT Value
parallel_reduce_int32ptr_int64_t_simple(const std::int32_t* begin, const std::int32_t* end, Value init, const Func &func, const Reduction &reduction) {
    const void *const lf = static_cast<const void *>(&func);
    const void *const rf = static_cast<const void *>(&reduction);

    return _onedal_parallel_reduce_int32ptr_int64_simple(begin,
                                         end,
                                         init,
                                         lf,
                                         parallel_reduce_loop_int32ptr_int64<Func>,
                                         rf,
                                         parallel_reduce_reduction_int64<Reduction>);
}

template <typename F>
ONEDAL_EXPORT void parallel_sort(F *begin_ptr, F *end_ptr) {
    throw unimplemented(dal::detail::error_messages::unimplemented_sorting_procedure());
}

#define ONEDAL_PARALLEL_SORT_SPECIALIZATION_DECL(TYPE) \
    template <>                                        \
    ONEDAL_EXPORT void parallel_sort(TYPE *begin_ptr, TYPE *end_ptr);

ONEDAL_PARALLEL_SORT_SPECIALIZATION_DECL(std::int32_t)
ONEDAL_PARALLEL_SORT_SPECIALIZATION_DECL(std::uint64_t)

#undef ONEDAL_PARALLEL_SORT_SPECIALIZATION_DECL

} // namespace oneapi::dal::detail
