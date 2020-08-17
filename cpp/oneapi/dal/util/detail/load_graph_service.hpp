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

#include <fstream>

#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/graph/detail/graph_container.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_array_graph_impl.hpp"
#include "oneapi/dal/graph/graph_common.hpp"
#include "oneapi/dal/graph/undirected_adjacency_array_graph.hpp"
#include "oneapi/dal/io/csv_data_source.hpp"
#include "oneapi/dal/io/load_graph_descriptor.hpp"
#include "services/daal_atomic_int.h"
#include "services/daal_memory.h"

namespace oneapi::dal::preview {
typedef void (*functype)(int i, const void *a);
}

extern "C" {
ONEAPI_DAL_EXPORT void _daal_threader_for_oneapi(int n,
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
inline ONEAPI_DAL_EXPORT void threader_for(size_t n, size_t threads_request, const F &lambda) {
    const void *a = static_cast<const void *>(&lambda);

    _daal_threader_for_oneapi((int)n, (int)threads_request, a, threader_func<F>);
}

ONEAPI_DAL_EXPORT int daal_string_to_int(const char *nptr, char **endptr);
} // namespace oneapi::dal::preview::load_graph::detail
