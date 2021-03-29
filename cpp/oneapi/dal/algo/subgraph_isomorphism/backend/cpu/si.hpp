#pragma once

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/solution.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/sorter.hpp"
#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/matching.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

template <typename T>
std::shared_ptr<T> make_shared_malloc(std::uint64_t elements_count) {
    T* ptr = static_cast<T*>(_mm_malloc(sizeof(T) * elements_count, 64));
    return std::shared_ptr<T>(ptr, _mm_free);
}

solution subgraph_isomorphism(const graph& pattern,
                              const graph& target,
                              const std::uint64_t control_flags = 0);

} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
