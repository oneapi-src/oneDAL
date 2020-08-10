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
#include "oneapi/dal/data/graph.hpp"
#include "oneapi/dal/data/table.hpp"

namespace oneapi::dal::preview {
namespace jaccard {

namespace detail {
class similarity_input_impl;
class similarity_result_impl;
} // namespace detail

class similarity_input {
public:
    similarity_input(const graph &g);
    const graph &get_graph() const;

private:
    dal::detail::pimpl<detail::similarity_input_impl> impl_;
};

using array_of_floats        = array<float>;
using array_of_pairs_uint32t = array<std::pair<std::uint32_t, std::uint32_t>>;

class similarity_result {
public:
    similarity_result(){};
    similarity_result(const array_of_floats &similarities,
                      const array_of_pairs_uint32t &vertex_pairs);
    array_of_floats get_jaccard_coefficients() const;
    array_of_pairs_uint32t get_vertex_pairs() const;

private:
    dal::detail::pimpl<detail::similarity_result_impl> impl_;
};
} // namespace jaccard
} // namespace oneapi::dal::preview
