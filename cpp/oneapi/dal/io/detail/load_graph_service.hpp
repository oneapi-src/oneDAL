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

#include "services/daal_atomic_int.h"
#include "services/daal_memory.h"

#include "oneapi/dal/graph/graph_common.hpp"
#include "oneapi/dal/detail/common.hpp"

extern "C" {
ONEAPI_DAL_EXPORT void _daal_parallel_sort_oneapi(int *begin_ptr, int *end_ptr);
}

namespace oneapi::dal::preview::load_graph::detail {
ONEAPI_DAL_EXPORT int daal_string_to_int(const char *nptr, char **endptr);

inline ONEAPI_DAL_EXPORT void parallel_sort(int *begin_ptr, int *end_ptr) {
    _daal_parallel_sort_oneapi(begin_ptr, end_ptr);
}
} // namespace oneapi::dal::preview::load_graph::detail
