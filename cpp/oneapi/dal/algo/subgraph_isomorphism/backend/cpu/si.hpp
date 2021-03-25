#pragma once

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/solution.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/sorter.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/matching.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

solution subgraph_isomorphism(const graph& pattern,
                              const graph& target,
                              const std::uint64_t control_flags = 0);

}
