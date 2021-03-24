#include "oneapi/dal/algo/subgraph_isomorphism/backend/cpu/graph_loader.hpp"
#include "debug.hpp"

#include <sstream>
#include <iostream>

namespace dal = oneapi::dal;

using namespace dal_experimental;

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

    PA(graph_data_storage.pbit_data->degree, vertex_count)
    // PA(graph_data_storage.plist_data->degree, vertex_count)
    pa("t._cols", t._cols.get_data(), (2 * t._edge_count));
    pa("t._rows", t._rows.get_data(), (vertex_count + 1));
    pa("t._degrees", t._degrees.get_data(), (vertex_count + 1));

    for (std::int64_t i = 0; i < vertex_count; i++) {
        auto degree = t._degrees[i];

        for (std::int64_t j = 0; j < degree; j++) {
            std::int64_t edge_attr = 0;
            std::int64_t vertex_1 = i;
            std::int64_t vertex_2 = t._cols[t._rows[i] + j];

            std::cout << "[" << vertex_1 << " " << vertex_2 << "]" << std::endl;

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
    pa_bit("graph_data_storage.pbit_data->data", graph_data_storage.pbit_data->data, vertex_count);
    pa_bit8("graph_data_storage.pbit_data->data", graph_data_storage.pbit_data->data, vertex_count);

    return;
}

graph_loader::graph_loader(const char *graph_file_name,
                           graph_file_type file_format,
                           graph_storage_scheme storage_scheme)
        : graph_loader() {
    std::ifstream graph_in_stream(graph_file_name);

    switch (file_format) {
        case grf: {
            read_grf_file(graph_in_stream, storage_scheme);
            break;
        }
        case gfd: {
            read_gfd_file(graph_in_stream, storage_scheme);
            break;
        }
        case mtx: {
            read_mtx_file(graph_in_stream, storage_scheme);
            break;
        }
    }
}

graph_loader::~graph_loader() {
    //need deep delete

    delete graph_data_storage.pbit_data;
    delete graph_data_storage.plist_data;
    graph_data_storage.pbit_data = nullptr;
    graph_data_storage.plist_data = nullptr;
}

std::int64_t graph_loader::read_scalar(std::istream &in) {
    read_line(in, line);

    int64_t scalar = 0;
    std::stringstream is(line);
    is >> scalar;

    return scalar;
}

graph_status graph_loader::read_mtx_header(std::istream &in,
                                           std::int64_t &row,
                                           std::int64_t &col,
                                           std::int64_t &element_count) {
    graph_status status = read_line(in, line);

    std::stringstream is(line);
    is >> row >> col >> element_count;

    return status;
}

graph_status graph_loader::read_edge(std::istream &in,
                                     std::int64_t &v1,
                                     std::int64_t &v2,
                                     std::int64_t &edge_attr) {
    graph_status status = read_line(in, line);
    if (status != ok) {
        return status;
    }

    std::stringstream is(line);
    edge_attr = -1;
    is >> v1 >> v2;
    if (!is.eof()) {
        is >> edge_attr;
    }

    return ok;
}

graph_status graph_loader::read_vertex_attribute(std::istream &in,
                                                 std::int64_t &vertex_id,
                                                 std::int64_t &vertex_attr) {
    char line[max_line_size + 1];
    graph_status status = read_line(in, line);
    if (status != ok) {
        return status;
    }

    std::stringstream is(line);
    is >> vertex_id >> vertex_attr;
    return ok;
}

graph_status graph_loader::read_line(std::istream &in, char *line) {
    char *character;
    do {
        *line = '\0';
        if (!in.good())
            return bad_arguments;
        in.getline(line, max_line_size);
        for (character = line; isspace(*character); character++) {
            ;
        }
    } while (*character == '\0' || *character == '#' || *character == '%');
    return ok;
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

graph_status graph_loader::read_grf_file(std::istream &in_stream,
                                         graph_storage_scheme storage_scheme) {
    has_edges_attribute = false;
    std::int64_t vertex_count = read_scalar(in_stream);
    if (vertex_count <= 0) {
        vertex_count = 0;
        return bad_arguments;
    }
    graph_status status = ok;

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

    std::int64_t *in_counter = nullptr;

    if (use_bit_representation) { // use bit vector
        graph_data_storage.pbit_data = new graph_input_bit_data(vertex_count);
    }
    else { // use adj list
        graph_data_storage.plist_data = new graph_input_list_data(vertex_count);
        in_counter =
            static_cast<std::int64_t *>(_mm_malloc(sizeof(std::int64_t) * vertex_count, 64));
    }

    std::int64_t vertex_id, vertex_attribute;
    for (std::int64_t i = 0; i < vertex_count; i++) {
        status = read_vertex_attribute(in_stream, vertex_id, vertex_attribute);
        if (status != ok) {
            return bad_arguments;
        }
        if (use_bit_representation) {
            graph_data_storage.pbit_data->attr[vertex_id] = vertex_attribute;
            graph_data_storage.pbit_data->degree[i] = 0;
        }
        else {
            graph_data_storage.plist_data->attr[vertex_id] = vertex_attribute;
            graph_data_storage.plist_data->degree[i] = 0;
            in_counter[i] = 0;
        }
    }

    for (std::int64_t i = 0; i < vertex_count; i++) {
        std::int64_t edge_count, vertex_1, vertex_2, edge_attr;
        edge_count = read_scalar(in_stream);
        if (use_bit_representation) {
            graph_data_storage.pbit_data->degree[i] += edge_count;
        }
        else {
            graph_data_storage.plist_data->degree[i] += edge_count;
            if (edge_count > 0) {
                graph_data_storage.plist_data->data[i] =
                    static_cast<std::int64_t *>(_mm_malloc(sizeof(std::int64_t) * edge_count, 64));
            }
            else {
                graph_data_storage.plist_data->data[i] = nullptr;
            }
        }

        for (std::int64_t j = 0; j < edge_count; j++) {
            status = read_edge(in_stream, vertex_1, vertex_2, edge_attr);
            if (status != ok) {
                return bad_arguments;
            }
            else {
                if (use_bit_representation) {
                    graph_data_storage.pbit_data->degree[vertex_2]++;
                    bit_vector::set_bit(graph_data_storage.pbit_data->data[vertex_1], vertex_2);
                    bit_vector::set_bit(graph_data_storage.pbit_data->data[vertex_2], vertex_1);
                    if (edge_attr >= 0 || has_edges_attribute) {
                        if (graph_data_storage.pbit_data->edges_attribute[i] == nullptr) {
                            graph_data_storage.pbit_data->edges_attribute[i] =
                                static_cast<std::int64_t *>(
                                    _mm_malloc(sizeof(std::int64_t) * edge_count, 64));
                            has_edges_attribute = true;
                        }
                        graph_data_storage.pbit_data->edges_attribute[i][j] = edge_attr;
                    }
                }
                else {
                    graph_data_storage.plist_data->degree[vertex_2]++;
                    graph_data_storage.plist_data->data[i][j] = vertex_2;
                    if (edge_attr >= 0 || has_edges_attribute) {
                        if (graph_data_storage.plist_data->edges_attribute[i] == nullptr) {
                            graph_data_storage.plist_data->edges_attribute[i] =
                                static_cast<std::int64_t *>(
                                    _mm_malloc(sizeof(std::int64_t) * edge_count, 64));
                            has_edges_attribute = true;
                        }
                        graph_data_storage.plist_data->edges_attribute[i][j] = edge_attr;
                    }
                }
            }
        }
    }

    if (!use_bit_representation) {
        for (std::int64_t i = 0; i < vertex_count; i++) {
            if (graph_data_storage.plist_data->degree[i] > 0) {
                graph_data_storage.plist_data->data[i] = static_cast<std::int64_t *>(
                    _mm_malloc(sizeof(std::int64_t) * graph_data_storage.plist_data->degree[i],
                               64));
            }
            else {
                graph_data_storage.plist_data->data[i] = nullptr;
            }
        }

        for (std::int64_t i = 0; i < vertex_count; i++) {
            for (std::int64_t j = 0; j < graph_data_storage.plist_data->degree[i]; j++) {
                auto v_id = graph_data_storage.plist_data->data[i][j];
                graph_data_storage.plist_data->data[v_id][in_counter[v_id]] = i;
                in_counter[v_id]++;
            }
        }
    }

    _mm_free(in_counter);
    in_counter = nullptr;

    // if (!use_bit_representation) {
    //     for (std::int64_t i = 0; i < vertex_count; i++) {
    //         for (std::int64_t j = 0; j < graph_data_storage.plist_data->degree[i]; j++) {
    //             auto v_id = graph_data_storage.plist_data->data[i][j];
    //             std::cout << v_id << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << "vertex_count = " << vertex_count << std::endl;
    // if (use_bit_representation) {
    //     for (std::int64_t i = 0; i < vertex_count; i++) {
    //         for (std::int64_t j = 0; j < vertex_count; j++) {
    //             std::cout << ((graph_data_storage.pbit_data->data[i][bit_vector::byte(j)] &
    //                            bit_vector::bit(j)) != 0);
    //         }
    //         std::cout << std::endl;
    //     }
    // }

    return ok;
}

graph_status graph_loader::read_gfd_file(std::istream &in_stream,
                                         graph_storage_scheme storage_scheme) {
    has_edges_attribute = false;
    graph_status status = ok;

    std::int64_t vertex_count = read_scalar(in_stream);
    if (vertex_count <= 0) {
        vertex_count = 0;
        return bad_arguments;
    }

    graph_input_data *input_data = new graph_input_bit_data(vertex_count);

    for (std::int64_t i = 0; i < vertex_count; i++) {
        input_data->attr[i] = read_scalar(in_stream);
        input_data->degree[i] = 0;
    }

    std::int64_t edge_count, vertex_1, vertex_2, edge_attr;
    edge_count = read_scalar(in_stream);

    switch (storage_scheme) {
        case auto_detect: {
            double density = graph::graph_density(vertex_count, edge_count);
            if (density < graph_storage_divider_by_density) { // use adj list
                use_bit_representation = false;
            }
            else {
                use_bit_representation = true;
            }
            break;
        }
        case list: {
            use_bit_representation = false;
            break;
        }
        default: {
            use_bit_representation = true;
            break;
        }
    }

    std::int64_t *src = nullptr;
    std::int64_t *dst = nullptr;
    std::int64_t *counter = nullptr;

    if (use_bit_representation) {
        graph_data_storage.pbit_data = new graph_input_bit_data(input_data);
    }
    else {
        graph_data_storage.plist_data = new graph_input_list_data(input_data);
        src = static_cast<std::int64_t *>(_mm_malloc(sizeof(std::int64_t) * edge_count, 64));
        dst = static_cast<std::int64_t *>(_mm_malloc(sizeof(std::int64_t) * edge_count, 64));
        counter = static_cast<std::int64_t *>(_mm_malloc(sizeof(std::int64_t) * vertex_count, 64));
    }
    delete input_data;
    input_data = nullptr;

    for (std::int64_t i = 0; i < edge_count; i++) {
        status = read_edge(in_stream, vertex_1, vertex_2, edge_attr);
        if (status != ok) {
            return bad_arguments;
        }
        if (use_bit_representation) {
            graph_data_storage.pbit_data->degree[vertex_1]++;
            graph_data_storage.pbit_data->degree[vertex_2]++;
            bit_vector::set_bit(graph_data_storage.pbit_data->data[vertex_1], vertex_2);
            bit_vector::set_bit(graph_data_storage.pbit_data->data[vertex_2], vertex_1);
        }
        else {
            graph_data_storage.plist_data->degree[vertex_1]++;
            graph_data_storage.plist_data->degree[vertex_2]++;
            src[i] = vertex_1;
            dst[i] = vertex_2;
        }
    }

    if (!use_bit_representation) {
        for (std::int64_t i = 0; i < graph_data_storage.plist_data->vertex_count; i++) {
            graph_data_storage.plist_data->data[i] = static_cast<std::int64_t *>(
                _mm_malloc(sizeof(std::int64_t) * graph_data_storage.plist_data->degree[i], 64));
            graph_data_storage.plist_data->data[i] = static_cast<std::int64_t *>(
                _mm_malloc(sizeof(std::int64_t) * graph_data_storage.plist_data->degree[i], 64));
            counter[i] = 0;
        }
        for (std::int64_t i = 0; i < edge_count; i++) {
            graph_data_storage.plist_data->data[src[i]][counter[src[i]]] = dst[i];
            graph_data_storage.plist_data->data[dst[i]][counter[dst[i]]] = src[i];
            counter[src[i]]++;
            counter[dst[i]]++;
        }

        _mm_free(src);
        src = nullptr;
        _mm_free(dst);
        dst = nullptr;
        _mm_free(counter);
        counter = nullptr;
    }

    return ok;
}

graph_status graph_loader::read_mtx_file(std::istream &in_stream,
                                         graph_storage_scheme storage_scheme) {
    std::int64_t vertex_1;
    std::int64_t vertex_2;
    std::int64_t edge_attr;
    std::int64_t edge_count;
    graph_status status = bad_arguments;

    status = read_mtx_header(in_stream, vertex_1, vertex_2, edge_count);
    if (status != ok || vertex_1 != vertex_2) {
        return status;
    }

    std::int64_t *src = nullptr;
    std::int64_t *dst = nullptr;
    std::int64_t *in_counter = nullptr;
    std::int64_t *out_counter = nullptr;

    switch (storage_scheme) {
        case auto_detect: {
            double density = graph::graph_density(vertex_1, edge_count);
            if (density < graph_storage_divider_by_density) { // use adj list
                use_bit_representation = false;
            }
            else {
                use_bit_representation = true;
            }
            break;
        }
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
        graph_data_storage.pbit_data = new graph_input_bit_data(vertex_1);
    }
    else { // use adj list
        graph_data_storage.plist_data = new graph_input_list_data(vertex_1);
        src = static_cast<std::int64_t *>(_mm_malloc(sizeof(std::int64_t) * edge_count, 64));
        dst = static_cast<std::int64_t *>(_mm_malloc(sizeof(std::int64_t) * edge_count, 64));
        in_counter = static_cast<std::int64_t *>(_mm_malloc(sizeof(std::int64_t) * vertex_1, 64));
        out_counter = static_cast<std::int64_t *>(_mm_malloc(sizeof(std::int64_t) * vertex_1, 64));
    }

    for (std::int64_t i = 0; i < edge_count; i++) {
        status = read_edge(in_stream, vertex_1, vertex_2, edge_attr);
        vertex_1--;
        vertex_2--;
        if (status != ok) {
            return bad_arguments;
        }

        if (use_bit_representation) {
            graph_data_storage.pbit_data->degree[vertex_1]++;
            graph_data_storage.pbit_data->degree[vertex_2]++;
            bit_vector::set_bit(graph_data_storage.pbit_data->data[vertex_1], vertex_2);
            bit_vector::set_bit(graph_data_storage.pbit_data->data[vertex_2], vertex_1);
        }
        else {
            graph_data_storage.plist_data->degree[vertex_1]++;
            graph_data_storage.plist_data->degree[vertex_2]++;
            src[i] = vertex_1;
            dst[i] = vertex_2;
        }
    }

    if (!use_bit_representation) {
        for (std::int64_t i = 0; i < graph_data_storage.plist_data->vertex_count; i++) {
            graph_data_storage.plist_data->data[i] = static_cast<std::int64_t *>(
                _mm_malloc(sizeof(std::int64_t) * graph_data_storage.plist_data->degree[i], 64));
            graph_data_storage.plist_data->data[i] = static_cast<std::int64_t *>(
                _mm_malloc(sizeof(std::int64_t) * graph_data_storage.plist_data->degree[i], 64));
            out_counter[i] = 0;
            in_counter[i] = 0;
        }
        for (std::int64_t i = 0; i < edge_count; i++) {
            graph_data_storage.plist_data->data[src[i]][out_counter[src[i]]] = dst[i];
            graph_data_storage.plist_data->data[dst[i]][in_counter[dst[i]]] = src[i];
            out_counter[src[i]]++;
            in_counter[dst[i]]++;
        }

        _mm_free(src);
        src = nullptr;
        _mm_free(dst);
        dst = nullptr;
        _mm_free(in_counter);
        in_counter = nullptr;
        _mm_free(out_counter);
        out_counter = nullptr;
    }

    return status;
}