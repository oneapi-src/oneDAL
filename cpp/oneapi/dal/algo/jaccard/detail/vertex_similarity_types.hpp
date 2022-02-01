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

/// @file
/// Contains the definition of the input and output for Jaccard Similarity
/// algorithm

#pragma once

#include "oneapi/dal/algo/jaccard/common.hpp"

namespace oneapi::dal::preview::jaccard::detail {

class vertex_similarity_result_impl;

template <typename Graph, typename Task>
class vertex_similarity_input_impl : public base {
public:
    vertex_similarity_input_impl(const Graph& g, caching_builder& builder_input)
            : graph_data(g),
              builder(builder_input) {}

    const Graph& graph_data;
    caching_builder& builder;
};

} // namespace oneapi::dal::preview::jaccard::detail
