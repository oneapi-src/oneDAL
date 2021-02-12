#include "oneapi/dal/test/engine/common.hpp"

#include "oneapi/dal/graph/service_functions.hpp"
#include "oneapi/dal/graph/undirected_adjacency_vector_graph.hpp"
#include "oneapi/dal/io/graph_csv_data_source.hpp"
#include "oneapi/dal/io/load_graph.hpp"

namespace oneapi::dal::graph::test {

namespace dal = oneapi::dal;

using my_graph_type = dal::preview::undirected_adjacency_vector_graph<>;
const dal::preview::load_graph::descriptor<dal::preview::edge_list<int32_t>, my_graph_type>
    desc;

auto get_data_source (const std::string& name) {
	const std::string name_path = "cpp/oneapi/dal/graph/test_data/" + name;
	const dal::preview::graph_csv_data_source ds(name_path);
	return ds;
}

auto check_neighbors (const std::string& name, 
	const dal::preview::vertex_size_type<my_graph_type> *correct_neigh,
	const dal::preview::vertex_size_type<my_graph_type> vertices_num) {
	int correct_neigh_num = 0;
	dal::preview::vertex_size_type<my_graph_type> neigh_index = 0;
	const auto my_graph = dal::preview::load_graph::load(desc, get_data_source(name));
	for (dal::preview::vertex_size_type<my_graph_type> j = 0; j < vertices_num; j++) {
		const auto neigh = dal::preview::get_vertex_neighbors(my_graph, j);
		for (auto i = neigh.first; i != neigh.second; ++i) {
            if (*i == correct_neigh[neigh_index]) correct_neigh_num++;	
        	neigh_index++;
        }
    }
    return correct_neigh_num;
}

TEST_CASE("throws if input file is empty") {
	REQUIRE_THROWS_WITH(dal::preview::load_graph::load(desc, get_data_source("graph0.csv")),"Empty edge list");
}

TEST_CASE("throws if input file does not exist") {
	REQUIRE_THROWS_WITH(dal::preview::load_graph::load(desc, get_data_source("nonexistgraph.csv")),"File not found");
}
////////////////////////////////

TEST_CASE("get_vertex_count for the graph with 2 vertices") {
	const auto my_graph = dal::preview::load_graph::load(desc, get_data_source("graph1.csv"));
	REQUIRE(dal::preview::get_vertex_count(my_graph) == 2);
}

TEST_CASE("get_vertex_count for the graph with 5 vertices and 1 edge") {
	const auto my_graph = dal::preview::load_graph::load(desc, get_data_source("graph2.csv"));
	REQUIRE(dal::preview::get_vertex_count(my_graph) == 5);
}

TEST_CASE("get_vertex_count for the complete graph") {
	const auto my_graph = dal::preview::load_graph::load(desc, get_data_source("graph3.csv"));
	REQUIRE(dal::preview::get_vertex_count(my_graph) == 7);
}

TEST_CASE("get_vertex_count for the example graph") {
	const auto my_graph = dal::preview::load_graph::load(desc, get_data_source("graph4.csv"));
	REQUIRE(dal::preview::get_vertex_count(my_graph) == 9);
}
////////////////////////////////

TEST_CASE("get_edge_count for the graph with 2 vertices") {
	const auto my_graph = dal::preview::load_graph::load(desc, get_data_source("graph1.csv"));
	REQUIRE(dal::preview::get_edge_count(my_graph) == 1);
}

TEST_CASE("get_edge_count for the graph with 5 vertices and 1 edge") {
	const auto my_graph = dal::preview::load_graph::load(desc, get_data_source("graph2.csv"));
	REQUIRE(dal::preview::get_edge_count(my_graph) == 1);
}

TEST_CASE("get_edge_count for the complete graph") {
	const auto my_graph = dal::preview::load_graph::load(desc, get_data_source("graph3.csv"));
	REQUIRE(dal::preview::get_edge_count(my_graph) == 21);
}  

TEST_CASE("get_edge_count for the example graph") {
	const auto my_graph = dal::preview::load_graph::load(desc, get_data_source("graph4.csv"));
	REQUIRE(dal::preview::get_edge_count(my_graph) == 12);
}    
////////////////////////////////

TEST_CASE("get_vertex_degree for the graph with 2 vertices") {
	int correct_degree_num = 0;
	const auto my_graph = dal::preview::load_graph::load(desc, get_data_source("graph1.csv"));
	for (dal::preview::vertex_size_type<my_graph_type> i=0; i<2; i++) {
			if (dal::preview::get_vertex_degree(my_graph, i) == 1) 
				correct_degree_num++;
	}
	REQUIRE(correct_degree_num == 2);
}

TEST_CASE("get_vertex_degree for the graph with 5 vertices and 1 edge") {
	int correct_degree_num = 0;
	const int correct_degrees[] = {1, 0, 0, 0, 1};
	const auto my_graph = dal::preview::load_graph::load(desc, get_data_source("graph2.csv"));
	for (dal::preview::vertex_size_type<my_graph_type> i = 0; i < 5; i++) {
		if (dal::preview::get_vertex_degree(my_graph, i) == correct_degrees[i]) 
			correct_degree_num++;
	}	
	REQUIRE(correct_degree_num == 5);			
}

TEST_CASE("get_vertex_degree for the complete graph") {	
	int correct_degree_num = 0;
	const auto my_graph = dal::preview::load_graph::load(desc, get_data_source("graph3.csv"));
	for (dal::preview::vertex_size_type<my_graph_type> i = 0; i < 7; i++) {
		if (dal::preview::get_vertex_degree(my_graph, i) == 6) 
			correct_degree_num++;
	}
	REQUIRE(correct_degree_num == 7);			
}

TEST_CASE("get_vertex_degree for the example graph") {
	int correct_degree_num = 0;	
	const int correct_degrees[] = {1, 4, 3, 1, 4, 0, 4, 3, 4};		
	const auto my_graph = dal::preview::load_graph::load(desc, get_data_source("graph4.csv"));
	for (dal::preview::vertex_size_type<my_graph_type> i = 0; i < 9; i++) {
		if (dal::preview::get_vertex_degree(my_graph, i) == correct_degrees[i]) 
			correct_degree_num++;
	}		
	REQUIRE(correct_degree_num == 9);			
}
////////////////////////////////

TEST_CASE("get_vertex_neighbors for the graph with 2 vertices") { 
	const dal::preview::vertex_size_type<my_graph_type> vertices_num = 2;
	const dal::preview::vertex_size_type<my_graph_type> correct_neigh[] = {1, 0};
	REQUIRE(check_neighbors ("graph1.csv", correct_neigh, vertices_num) == 2);
}

TEST_CASE("get_vertex_neighbors for the graph with 5 vertices and 1 edge") { 
	const dal::preview::vertex_size_type<my_graph_type> vertices_num = 5;
	const dal::preview::vertex_size_type<my_graph_type> correct_neigh[] = {4, 0};
	REQUIRE(check_neighbors ("graph2.csv", correct_neigh, vertices_num) == 2);
}	

TEST_CASE("get_vertex_neighbors for the complete graph") {
	int correct_neigh_num = 0;
	dal::preview::vertex_size_type<my_graph_type> neigh_vertex = 0;
	const auto my_graph = dal::preview::load_graph::load(desc, get_data_source("graph3.csv"));
	for (dal::preview::vertex_size_type<my_graph_type> j = 0; j < 7; j++) {
		const auto neigh = dal::preview::get_vertex_neighbors(my_graph, j);
		for (auto i = neigh.first; i != neigh.second; ++i) {
            if (j == neigh_vertex) neigh_vertex++;
            if (*i == neigh_vertex) correct_neigh_num++;
        	neigh_vertex++;
        }
        neigh_vertex = 0;
    }
	REQUIRE(correct_neigh_num == 42);	
}

TEST_CASE("get_vertex_neighbors for the example graph") { 
	const dal::preview::vertex_size_type<my_graph_type> vertices_num = 9;
	const dal::preview::vertex_size_type<my_graph_type> correct_neigh[] = {8, 2, 4, 6, 7, 1, 4, 6, 8, 1, 2, 6, 8, 1, 2, 4, 7, 1, 6, 8, 0, 3, 4, 7};
	REQUIRE(check_neighbors ("graph4.csv", correct_neigh, vertices_num) == 24);
}

} // namespace oneapi::dal::graph::test
