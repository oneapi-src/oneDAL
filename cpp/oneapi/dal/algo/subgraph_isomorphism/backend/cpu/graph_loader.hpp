#pragma once

#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph.hpp"
#include "oneapi/dal/graph/detail/undirected_adjacency_vector_graph_impl.hpp"

namespace dal = oneapi::dal;

#include <fstream>

namespace dal_experimental {

enum graph_storage_scheme { auto_detect, bit, list };

enum graph_file_type { grf, gfd, mtx };

class graph_loader {
public:
    graph_loader();
    graph_loader(const dal::preview::detail::topology<std::int32_t>& t,
                 graph_storage_scheme storage_scheme);
    graph_loader(const char* graph_file_name,
                 graph_file_type file_format = grf,
                 graph_storage_scheme storage_scheme = auto_detect);
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

    std::int64_t read_scalar(std::istream& in);
    graph_status read_mtx_header(std::istream& in,
                                 std::int64_t& row,
                                 std::int64_t& col,
                                 std::int64_t& element_count);
    graph_status read_line(std::istream& in, char* line);
    graph_status read_vertex_attribute(std::istream& in,
                                       std::int64_t& vertex_id,
                                       std::int64_t& vertex_attr);
    graph_status read_edge(std::istream& in,
                           std::int64_t& v1,
                           std::int64_t& v2,
                           std::int64_t& edge_attr);

    graph_status read_grf_file(std::istream& in_stream,
                               graph_storage_scheme storage_scheme = auto_detect);
    graph_status read_gfd_file(std::istream& in_stream,
                               graph_storage_scheme storage_scheme = auto_detect);
    graph_status read_mtx_file(std::istream& in_stream,
                               graph_storage_scheme storage_scheme = auto_detect);
};

} // namespace dal_experimental