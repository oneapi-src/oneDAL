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

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/backend/memory.hpp"
#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::preview::louvain::backend {

using namespace oneapi::dal::preview::detail;
using namespace oneapi::dal::preview::backend;

template <typename IndexType, typename EdgeValue>
class graph {
    using vertex_type = IndexType;
    using vertex_size_type = std::int64_t;
    using value_type = EdgeValue;
    using edge_type = std::int64_t;
    using edge_size_type = std::int64_t;

    using value_allocator_type = inner_alloc<value_type>;
    using vertex_allocator_type = inner_alloc<vertex_type>;
    using vertex_size_allocator_type = inner_alloc<vertex_size_type>;

    const std::int64_t initial_vertex_count = 0;
    const std::int64_t initial_edge_count = 0;

public:
    edge_type* rows;
    vertex_type* cols;
    value_type* vals;
    value_type* self_loops;

    std::int64_t vertex_count = 0;
    std::int64_t edge_count = 0;

    vertex_allocator_type vertex_allocator;
    vertex_size_allocator_type vertex_size_allocator;
    value_allocator_type value_allocator;
    byte_alloc_iface* alloc_ptr;

    graph(const dal::preview::detail::topology<std::int32_t>& t,
          const EdgeValue* t_vals,
          byte_alloc_iface* alloc_ptr_)
            : initial_vertex_count(t.get_vertex_count()),
              initial_edge_count(t.get_edge_count()),
              vertex_count(t.get_vertex_count()),
              edge_count(t.get_edge_count()),
              vertex_allocator(alloc_ptr_),
              vertex_size_allocator(alloc_ptr_),
              value_allocator(alloc_ptr_),
              alloc_ptr(alloc_ptr_) {
        rows = allocate(vertex_size_allocator, vertex_count + 1);
        cols = allocate(vertex_allocator, edge_count * 2);
        vals = allocate(value_allocator, edge_count * 2);
        self_loops = allocate(value_allocator, vertex_count);

        for (std::int64_t index = 0; index <= vertex_count; ++index) {
            this->rows[index] = t._rows_ptr[index];
        }
        for (std::int64_t index = 0; index < edge_count * 2; ++index) {
            cols[index] = t._cols_ptr[index];
            vals[index] = t_vals[index];
        }
        for (std::int64_t index = 0; index < vertex_count; ++index) {
            self_loops[index] = 0;
        }
    }
    ~graph() {
        deallocate(vertex_size_allocator, rows, initial_vertex_count + 1);
        deallocate(vertex_allocator, cols, initial_edge_count * 2);
        deallocate(value_allocator, vals, initial_edge_count * 2);
        deallocate(value_allocator, self_loops, initial_vertex_count);
    }
};

} // namespace oneapi::dal::preview::louvain::backend
