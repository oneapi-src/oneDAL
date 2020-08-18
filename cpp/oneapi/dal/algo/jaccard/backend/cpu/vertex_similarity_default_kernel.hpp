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

#include "oneapi/dal/algo/jaccard/common.hpp"
#include "oneapi/dal/algo/jaccard/vertex_similarity_types.hpp"
#include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/detail/policy.hpp"

#include <memory>

namespace oneapi::dal::preview {
namespace jaccard {
namespace detail {

template <typename Graph, typename Cpu>
vertex_similarity_result call_jaccard_default_kernel(const descriptor_base &desc,
                                                     vertex_similarity_input<Graph> &input);

#define INSTANTIATE(cpu)                                                  \
    template vertex_similarity_result                                     \
    call_jaccard_default_kernel<undirected_adjacency_array_graph<>, cpu>( \
        const descriptor_base &desc,                                      \
        vertex_similarity_input<undirected_adjacency_array_graph<>> &input);
} // namespace detail
} // namespace jaccard
} // namespace oneapi::dal::preview
