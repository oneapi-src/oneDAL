#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/matching.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

namespace dal = oneapi::dal;

template class matching_engine<__CPU_TAG__>;
template class engine_bundle<__CPU_TAG__>;
} // namespace oneapi::dal::preview::subgraph_isomorphism::detail
