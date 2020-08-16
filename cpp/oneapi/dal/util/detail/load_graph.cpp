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

#include "oneapi/dal/util/detail/load_graph.hpp"
#include "src/externals/service_service.h"

namespace oneapi::dal::preview::load_graph::detail {

ONEAPI_DAL_EXPORT int daal_string_to_int(const char* nptr, char** endptr) {
    return daal::internal::Service<>::serv_string_to_int(nptr, endptr);
}
} // namespace oneapi::dal::preview::load_graph::detail

#if defined(__DO_TBB_LAYER__)
    #define TBB_PREVIEW_GLOBAL_CONTROL 1
    #define TBB_PREVIEW_TASK_ARENA     1

    #include <stdlib.h> // malloc and free
    #include <tbb/global_control.h>
    #include <tbb/scalable_allocator.h>
    #include <tbb/spin_mutex.h>
    #include <tbb/task_arena.h>
    #include <tbb/tbb.h>
    #include "services/daal_atomic_int.h"

    #if defined(TBB_INTERFACE_VERSION) && TBB_INTERFACE_VERSION >= 12002
        #include <tbb/task.h>
    #endif

using namespace daal::services;
#else
    #include "src/externals/service_service.h"
#endif

ONEAPI_DAL_EXPORT void _daal_threader_for_oneapi(int n,
                                                 int threads_request,
                                                 const void* a,
                                                 oneapi::dal::preview::functype func) {
#if defined(__DO_TBB_LAYER__)
    tbb::parallel_for(tbb::blocked_range<int>(0, n, 1), [&](tbb::blocked_range<int> r) {
        int i;
        for (i = r.begin(); i < r.end(); i++) {
            func(i, a);
        }
    });
#elif defined(__DO_SEQ_LAYER__)
    int i;
    for (i = 0; i < n; i++) {
        func(i, a);
    }
#endif
}