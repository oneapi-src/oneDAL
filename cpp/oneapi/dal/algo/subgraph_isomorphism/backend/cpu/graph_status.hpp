#pragma once

namespace oneapi::dal::preview::subgraph_isomorphism::detail {

enum graph_status {
    ok = 0, /*!< No error found*/
    bad_arguments = -5, /*!< Bad argument(s) passed*/
    bad_allocation = -11, /*!< Memory allocation error*/
};

enum edge_direction {
    none = 0, /*!< No edge*/
    both = 1 /*!< Edge exist */
};

} // namespace oneapi::dal::preview::subgraph_isomorphism::detail