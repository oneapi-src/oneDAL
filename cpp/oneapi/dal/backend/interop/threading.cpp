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

ONEDAL_EXPORT void _onedal_threader_for(std::int32_t n,
                                        std::int32_t threads_request,
                                        const void *a,
                                        oneapi::dal::preview::functype func) {
    _daal_threader_for(n, threads_request, a, static_cast<daal::functype>(func));
}

ONEDAL_EXPORT std::int32_t _onedal_parallel_reduce(
    std::int32_t n,
    std::int32_t init,
    const void *a,
    oneapi::dal::preview::loop_functype loop_func,
    const void *b,
    oneapi::dal::preview::reduce_function reduction_func) {
    static_assert(sizeof(int) == sizeof(std::int32_t));
    return _daal_parallel_reduce((int)n,
                                 (int)init,
                                 a,
                                 static_cast<daal::loop_functype>(loop_func),
                                 b,
                                 static_cast<daal::reduce_function>(reduction_func));
}

namespace oneapi::dal::detail {

#define ONEDAL_PARALLEL_SORT_SPECIALIZATION(TYPE, DAALTYPE, NAMESUFFIX)               \
    template <>                                                                       \
    ONEDAL_EXPORT void parallel_sort(TYPE *begin_ptr, TYPE *end_ptr) {                \
        static_assert(sizeof(TYPE) == sizeof(DAALTYPE));                              \
        _daal_parallel_sort_##NAMESUFFIX((DAALTYPE *)begin_ptr, (DAALTYPE *)end_ptr); \
    }

ONEDAL_PARALLEL_SORT_SPECIALIZATION(std::int32_t, int, int32)
ONEDAL_PARALLEL_SORT_SPECIALIZATION(std::uint64_t, size_t, uint64)

#undef ONEDAL_PARALLEL_SORT_SPECIALIZATION

} // namespace oneapi::dal::detail
