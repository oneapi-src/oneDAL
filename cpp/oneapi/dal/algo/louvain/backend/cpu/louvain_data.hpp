/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/backend/primitives/rng/rng_engine.hpp"

namespace oneapi::dal::preview::louvain::backend {
using namespace oneapi::dal::preview::detail;
using namespace oneapi::dal::preview::backend;
using namespace oneapi::dal::backend::primitives;

template <typename IndexType, typename EdgeValue>
struct louvain_data {
    using value_type = EdgeValue;
    using vertex_type = std::int32_t;
    using vertex_size_type = std::int64_t;

    using value_allocator_type = inner_alloc<value_type>;
    using vertex_allocator_type = inner_alloc<vertex_type>;
    using vertex_size_allocator_type = inner_alloc<vertex_size_type>;

    using v1v_t = vector_container<vertex_type, vertex_allocator_type>;
    using v1a_t = inner_alloc<v1v_t>;
    using v2v_t = vector_container<v1v_t, v1a_t>;

    louvain_data() = delete;
    louvain_data(std::int64_t vertex_count,
                 std::int64_t edge_count,
                 value_allocator_type& value_allocator,
                 vertex_allocator_type& vertex_allocator,
                 vertex_size_allocator_type& vertex_size_allocator,
                 v1a_t& v1a)
            : c2v(vertex_count, v1a),
              m(0) {
        k_shared_ptr = value_allocator.make_shared_memory(vertex_count);
        tot_shared_ptr = value_allocator.make_shared_memory(vertex_count);
        k_vertex_to_shared_ptr = value_allocator.make_shared_memory(vertex_count);
        neighboring_communities_shared_ptr = vertex_allocator.make_shared_memory(vertex_count);
        random_order_shared_ptr = vertex_allocator.make_shared_memory(vertex_count);
        empty_community_shared_ptr = vertex_allocator.make_shared_memory(vertex_count);
        community_size_shared_ptr = vertex_size_allocator.make_shared_memory(vertex_count);

        k_c_shared_ptr = value_allocator.make_shared_memory(vertex_count);
        local_self_loops_shared_ptr = value_allocator.make_shared_memory(vertex_count);

        weights_shared_ptr = value_allocator.make_shared_memory(vertex_count);
        c_self_loops_shared_ptr = value_allocator.make_shared_memory(vertex_count);
        c_neighbors_shared_ptr = vertex_allocator.make_shared_memory(vertex_count);
        c_rows_shared_ptr = vertex_allocator.make_shared_memory(vertex_count + 1);

        c_vals_shared_ptr = value_allocator.make_shared_memory(edge_count * 2);
        c_cols_shared_ptr = vertex_allocator.make_shared_memory(edge_count * 2);
        index_shared_ptr = vertex_allocator.make_shared_memory(vertex_count);

        k = k_shared_ptr.get();
        tot = tot_shared_ptr.get();
        k_vertex_to = k_vertex_to_shared_ptr.get();
        neighboring_communities = neighboring_communities_shared_ptr.get();
        random_order = random_order_shared_ptr.get();
        empty_community = empty_community_shared_ptr.get();
        community_size = community_size_shared_ptr.get();

        k_c = k_c_shared_ptr.get();
        local_self_loops = local_self_loops_shared_ptr.get();

        weights = weights_shared_ptr.get();
        c_self_loops = c_self_loops_shared_ptr.get();
        c_neighbors = c_neighbors_shared_ptr.get();
        c_rows = c_rows_shared_ptr.get();

        c_vals = c_vals_shared_ptr.get();
        c_cols = c_cols_shared_ptr.get();
        index = index_shared_ptr.get();
    }

    oneapi::dal::detail::shared<value_type> k_shared_ptr;
    oneapi::dal::detail::shared<value_type> tot_shared_ptr;
    oneapi::dal::detail::shared<value_type> k_vertex_to_shared_ptr;
    oneapi::dal::detail::shared<vertex_type> neighboring_communities_shared_ptr;
    oneapi::dal::detail::shared<vertex_type> random_order_shared_ptr;
    oneapi::dal::detail::shared<vertex_type> empty_community_shared_ptr;
    oneapi::dal::detail::shared<vertex_size_type> community_size_shared_ptr;

    oneapi::dal::detail::shared<value_type> k_c_shared_ptr;
    oneapi::dal::detail::shared<value_type> local_self_loops_shared_ptr;

    oneapi::dal::detail::shared<value_type> weights_shared_ptr;
    oneapi::dal::detail::shared<value_type> c_self_loops_shared_ptr;
    oneapi::dal::detail::shared<vertex_type> c_neighbors_shared_ptr;
    oneapi::dal::detail::shared<vertex_type> c_rows_shared_ptr;

    oneapi::dal::detail::shared<value_type> c_vals_shared_ptr;
    oneapi::dal::detail::shared<vertex_type> c_cols_shared_ptr;
    oneapi::dal::detail::shared<vertex_type> index_shared_ptr;

    value_type* k;
    value_type* tot;
    value_type* k_vertex_to;
    vertex_type* neighboring_communities;
    vertex_type* random_order;
    vertex_type* empty_community;
    vertex_size_type* community_size;

    value_type* k_c;
    value_type* local_self_loops;

    value_type* weights;
    value_type* c_self_loops;
    vertex_type* c_neighbors;
    vertex_type* c_rows;

    value_type* c_vals;
    vertex_type* c_cols;
    vertex_type* index;

    v2v_t c2v;
    value_type m;

    engine eng;
    rng<std::int32_t> rn_gen;
};

} // namespace oneapi::dal::preview::louvain::backend
