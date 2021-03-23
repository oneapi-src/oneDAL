#pragma once

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/solution.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/sorter.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/matching.hpp"

namespace oneapi::dal::preview {
namespace subgraph_isomorphism {
namespace detail {

dal_experimental::solution subgraph_isomorphism(const dal_experimental::graph& pattern,
                                                const dal_experimental::graph& target,
                                                const std::uint64_t control_flags = 0);

}
} // namespace subgraph_isomorphism
} // namespace oneapi::dal::preview