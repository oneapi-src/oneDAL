/*******************************************************************************
 * Copyright 2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "oneapi/dal/algo/jaccard.hpp"
#include <iostream>
#include "oneapi/dal/data/graph.hpp"
#include "oneapi/dal/data/table.hpp"
#include "oneapi/dal/util/csv_data_source.hpp"
#include "oneapi/dal/util/load_graph.hpp"

using namespace oneapi::dal;
using namespace oneapi::dal::preview;

int main(int argc, char **argv) {
    if (argc < 2)
        return 0;
    std::string filename = argv[1];
    csv_data_source ds(filename);
    edge_list_to_csr_descriptor d;
    auto my_graph = load_graph<graph>(d, ds);

    const auto jaccard_desc_default =
        jaccard::descriptor<>().set_row_begin(0).set_row_end(2).set_column_begin(0).set_column_end(
            3);

    auto result_default = vertex_similarity(jaccard_desc_default, my_graph);

    array<float> jaccard = result_default.get_jaccard_coefficients();
    array<std::pair<std::uint32_t, std::uint32_t>> vertex_pairs = result_default.get_vertex_pairs();

    std::cout << "Number of non-zero coefficients in block =" << jaccard.get_size() << std::endl;
    std::cout << "Jaccard indices:" << std::endl;
    for (auto i = 0; i < jaccard.get_size(); ++i) {
        std::cout << "Pair: (" << vertex_pairs[i].first << "," << vertex_pairs[i].second
                  << ")\tcoefficient: " << jaccard[i] << std::endl;
    }
}
