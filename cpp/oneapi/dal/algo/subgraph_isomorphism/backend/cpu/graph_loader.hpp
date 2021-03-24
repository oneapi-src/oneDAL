#pragma once

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"

namespace dal = oneapi::dal;

#include <fstream>

namespace dal_experimental {

enum graph_storage_scheme { auto_detect, bit, list };

class graph_loader {
public:
    graph_loader();
    graph_loader(const dal::preview::detail::topology<std::int32_t>& t,
                 graph_storage_scheme storage_scheme);
    virtual ~graph_loader();

    const graph_input_bit_data* get_bit_data() const;
    const graph_input_list_data* get_list_data() const;
    const graph_data* get_graph_data() const;

private:
    graph_data graph_data_storage;

    static const std::int64_t max_line_size = 512;
    char line[max_line_size + 1];
    bool has_edges_attribute;
    bool use_bit_representation;
};

} // namespace dal_experimental