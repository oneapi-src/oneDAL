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

#include "oneapi/dal/detail/threading.hpp"
#include "src/threading/threading.h"

ONEDAL_EXPORT int _onedal_threader_get_max_threads() {
    return _daal_threader_get_max_threads();
}

ONEDAL_EXPORT int _onedal_threader_get_current_thread_index() {
    return _daal_threader_get_current_thread_index();
}

ONEDAL_EXPORT void _onedal_threader_for(std::int32_t n,
                                        std::int32_t threads_request,
                                        const void *a,
                                        oneapi::dal::preview::functype func) {
    _daal_threader_for(n, threads_request, a, static_cast<daal::functype>(func));
}

ONEDAL_EXPORT void _onedal_threader_for_simple(std::int32_t n,
                                        std::int32_t threads_request,
                                        const void *a,
                                        oneapi::dal::preview::functype func) {
    _daal_threader_for_simple(n, threads_request, a, static_cast<daal::functype>(func));
}

ONEDAL_EXPORT void _onedal_threader_for_int32ptr(const std::int32_t* begin,
                                        const std::int32_t* end,
                                        const void *a,
                                        oneapi::dal::preview::functype_int32ptr func) {
    _daal_threader_for_int32ptr(begin, end, a, static_cast<daal::functype_int32ptr>(func));
}

ONEDAL_EXPORT std::int64_t _onedal_parallel_reduce_size_t_int64(
    size_t n,
    std::int64_t init,
    const void *a,
    oneapi::dal::preview::loop_functype_size_t_int64 loop_func,
    const void *b,
    oneapi::dal::preview::reduction_functype_int64 reduction_func) {
    return _daal_parallel_reduce_size_t_int64(n,
                                       (std::int64_t)init,
                                       a,
                                       static_cast<daal::loop_functype_size_t_int64>(loop_func),
                                       b,
                                       static_cast<daal::reduction_functype_int64>(reduction_func));
}

ONEDAL_EXPORT std::int64_t _onedal_parallel_reduce_size_t_int64_simple(
    size_t n,
    std::int64_t init,
    const void *a,
    oneapi::dal::preview::loop_functype_size_t_int64 loop_func,
    const void *b,
    oneapi::dal::preview::reduction_functype_int64 reduction_func) {
    return _daal_parallel_reduce_size_t_int64_simple(n,
                                       (std::int64_t)init,
                                       a,
                                       static_cast<daal::loop_functype_size_t_int64>(loop_func),
                                       b,
                                       static_cast<daal::reduction_functype_int64>(reduction_func));
}

ONEDAL_EXPORT std::int64_t _onedal_parallel_reduce_int32ptr_int64_simple(
    const int32_t* begin,
    const int32_t* end,
    std::int64_t init,
    const void *a,
    oneapi::dal::preview::loop_functype_int32ptr_int64 loop_func,
    const void *b,
    oneapi::dal::preview::reduction_functype_int64 reduction_func) {
    return _daal_parallel_reduce_int32ptr_int64_simple(begin,
                                       end,
                                       (std::int64_t)init,
                                       a,
                                       static_cast<daal::loop_functype_int32ptr_int64>(loop_func),
                                       b,
                                       static_cast<daal::reduction_functype_int64>(reduction_func));
}

namespace oneapi::dal::detail {

typedef std::pair<std::int32_t, size_t> pair_int32_t_size_t;

#define ONEDAL_PARALLEL_SORT_SPECIALIZATION(TYPE, DAALTYPE, NAMESUFFIX)               \
    template <>                                                                       \
    ONEDAL_EXPORT void parallel_sort(TYPE *begin_ptr, TYPE *end_ptr) {                \
        static_assert(sizeof(TYPE) == sizeof(DAALTYPE));                              \
        _daal_parallel_sort_##NAMESUFFIX((DAALTYPE *)begin_ptr, (DAALTYPE *)end_ptr); \
    }

ONEDAL_PARALLEL_SORT_SPECIALIZATION(std::int32_t, int, int32)
ONEDAL_PARALLEL_SORT_SPECIALIZATION(std::uint64_t, size_t, uint64)
ONEDAL_PARALLEL_SORT_SPECIALIZATION(pair_int32_t_size_t, daal::IdxValType<int>, pair_int32_uint64)

#undef ONEDAL_PARALLEL_SORT_SPECIALIZATION

} // namespace oneapi::dal::detail
