#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/si.hpp"

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

template solution si<__CPU_TAG__>(const graph& pattern,
                                  const graph& target,
                                  kind isomorphism_kind,
                                  byte_alloc_iface* alloc_ptr);

template subgraph_isomorphism::graph_matching_result si_call_kernel<__CPU_TAG__>(
    const kind& si_kind,
    byte_alloc_iface* alloc_ptr,
    const dal::preview::detail::topology<std::int32_t>& t_data,
    const dal::preview::detail::topology<std::int32_t>& p_data,
    const std::int64_t* vv_t,
    const std::int64_t* vv_p);

} // namespace oneapi::dal::preview::subgraph_isomorphism::backend
