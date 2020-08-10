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

#include "oneapi/dal/algo/jaccard/backend/cpu/select_kernel.hpp"

namespace oneapi::dal::preview {
namespace jaccard {
namespace detail {

template <typename Graph>
extern similarity_result call_jaccard_block_kernel(const descriptor_base &desc,
                                                   const similarity_input<Graph> &input);

template <typename Float, typename Method, typename Graph>
similarity_result backend_default<Float, Method, Graph>::operator()(
    const descriptor_base &desc,
    const similarity_input<Graph> &input) {
    return call_jaccard_block_kernel<Graph>(desc, input);
}

template <>
dal::detail::pimpl<backend_base<undirected_adjacency_array<> &>>
get_backend<float, method::by_default, undirected_adjacency_array<> &>(
    const descriptor_base &desc,
    const similarity_input<undirected_adjacency_array<> &> &input) {
    return dal::detail::pimpl<backend_base<undirected_adjacency_array<> &>>(
        new backend_default<float, method::by_default, undirected_adjacency_array<> &>);
}

} // namespace detail
} // namespace jaccard
} // namespace oneapi::dal::preview
