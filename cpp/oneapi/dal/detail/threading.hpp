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

namespace oneapi::dal::preview {
typedef void (*functype)(int i, const void *a);
}

extern "C" {
ONEDAL_EXPORT void _daal_threader_for_oneapi(int n,
                                             int threads_request,
                                             const void *a,
                                             oneapi::dal::preview::functype func);
}

namespace oneapi::dal::preview::load_graph::detail {
template <typename F>
inline void threader_func(int i, const void *a) {
    const F &lambda = *static_cast<const F *>(a);
    lambda(i);
}

template <typename F>
inline ONEDAL_EXPORT void threader_for(size_t n, size_t threads_request, const F &lambda) {
    const void *a = static_cast<const void *>(&lambda);

    _daal_threader_for_oneapi((int)n, (int)threads_request, a, threader_func<F>);
}
} // namespace oneapi::dal::preview::load_graph::detail
