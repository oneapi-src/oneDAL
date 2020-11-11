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

#include "oneapi/dal/detail/threading.hpp"
#include "src/threading/threading.h"

ONEDAL_EXPORT void _onedal_threader_for(int n,
                                        int threads_request,
                                        const void *a,
                                        oneapi::dal::preview::functype func) {
    _daal_threader_for(n, threads_request, a, static_cast<daal::functype>(func));
}

#define ONEDAL_PARALLEL_SORT_IMPL(TYPE)                                               \
    ONEDAL_EXPORT void _onedal_parallel_sort_##TYPE(TYPE *begin_ptr, TYPE *end_ptr) { \
        _daal_parallel_sort_##TYPE(begin_ptr, end_ptr);                               \
    }

ONEDAL_PARALLEL_SORT_IMPL(int)
ONEDAL_PARALLEL_SORT_IMPL(size_t)

#undef ONEDAL_PARALLEL_SORT_IMPL
