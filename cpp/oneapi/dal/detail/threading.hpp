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

#include "oneapi/dal/detail/common.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::preview {
typedef void (*functype)(std::int32_t i, const void *a);
}

extern "C" {
ONEDAL_EXPORT void _onedal_threader_for(std::int32_t n,
                                        std::int32_t threads_request,
                                        const void *a,
                                        oneapi::dal::preview::functype func);
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
