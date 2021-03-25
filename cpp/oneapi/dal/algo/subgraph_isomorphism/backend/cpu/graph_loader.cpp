#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph_loader.hpp"
#include "debug.hpp"

namespace dal = oneapi::dal;

namespace oneapi::dal::preview::subgraph_isomorphism::backend {

graph_loader::graph_loader() {
    graph_data_storage.pbit_data = nullptr;
    graph_data_storage.plist_data = nullptr;
}

graph_loader::graph_loader(const dal::preview::detail::topology<std::int32_t> &t,
                           graph_storage_scheme storage_scheme) {
    has_edges_attribute = false;
    std::int64_t vertex_count = t._vertex_count;
    if (vertex_count <= 0) {
        vertex_count = 0;
        return;
    }

    switch (storage_scheme) {
        case list: {
            use_bit_representation = false;
            break;
        }
        default: {
            use_bit_representation = true;
            break;
        }
    }

    if (use_bit_representation) { // use bit vector
        graph_data_storage.pbit_data = new graph_input_bit_data(vertex_count);
    }
    else { // use adj list
        graph_data_storage.plist_data = new graph_input_list_data(vertex_count);
    }

    std::int64_t vertex_id, vertex_attribute;
    for (std::int64_t i = 0; i < vertex_count; i++) {
        auto degree = t._degrees[i];
        if (use_bit_representation) {
            graph_data_storage.pbit_data->degree[i] = degree;
        }
        else {
            graph_data_storage.plist_data->degree[i] = degree;
            if (degree > 0) {
                graph_data_storage.plist_data->data[i] =
                    static_cast<std::int64_t *>(_mm_malloc(sizeof(std::int64_t) * degree, 64));
            }
            else {
                graph_data_storage.plist_data->data[i] = nullptr;
            }
        }
    }

    for (std::int64_t i = 0; i < vertex_count; i++) {
        auto degree = t._degrees[i];

        for (std::int64_t j = 0; j < degree; j++) {
            std::int64_t edge_attr = 0;
            std::int64_t vertex_1 = i;
            std::int64_t vertex_2 = t._cols[t._rows[i] + j];

            if (use_bit_representation) {
                bit_vector::set_bit(graph_data_storage.pbit_data->data[vertex_1], vertex_2);
                bit_vector::set_bit(graph_data_storage.pbit_data->data[vertex_2], vertex_1);
                if (edge_attr >= 0 || has_edges_attribute) {
                    if (graph_data_storage.pbit_data->edges_attribute[i] == nullptr) {
                        graph_data_storage.pbit_data->edges_attribute[i] =
                            static_cast<std::int64_t *>(
                                _mm_malloc(sizeof(std::int64_t) * degree, 64));
                        has_edges_attribute = true;
                    }
                    graph_data_storage.pbit_data->edges_attribute[i][j] = edge_attr;
                }
            }
            else {
                graph_data_storage.plist_data->data[i][j] = vertex_2;
                if (edge_attr >= 0 || has_edges_attribute) {
                    if (graph_data_storage.plist_data->edges_attribute[i] == nullptr) {
                        graph_data_storage.plist_data->edges_attribute[i] =
                            static_cast<std::int64_t *>(
                                _mm_malloc(sizeof(std::int64_t) * degree, 64));
                        has_edges_attribute = true;
                    }
                    graph_data_storage.plist_data->edges_attribute[i][j] = edge_attr;
                }
            }
        }
    }
    return;
}

graph_loader::~graph_loader() {
    //need deep delete

    delete graph_data_storage.pbit_data;
    delete graph_data_storage.plist_data;
    graph_data_storage.pbit_data = nullptr;
    graph_data_storage.plist_data = nullptr;
}

const graph_data *graph_loader::get_graph_data() const {
    return &graph_data_storage;
}

const graph_input_bit_data *graph_loader::get_bit_data() const {
    return graph_data_storage.pbit_data;
}

const graph_input_list_data *graph_loader::get_list_data() const {
    return graph_data_storage.plist_data;
}
} // namespace oneapi::dal::preview::subgraph_isomorphism::backend