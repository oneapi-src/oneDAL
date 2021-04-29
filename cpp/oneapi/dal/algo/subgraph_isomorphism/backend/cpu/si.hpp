#pragma once

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/solution.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/common.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

solution si(const graph& pattern,
            const graph& target,
            kind isomorphism_kind,
            byte_alloc_iface* alloc_ptr);

} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
