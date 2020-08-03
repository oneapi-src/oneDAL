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

#include "oneapi/dal/util/load_graph.hpp"

#include <fstream>

namespace oneapi::dal::preview {
namespace detail {
template <>
edge_list<std::int64_t> load(const std::string &name) {
    using int_t = std::int64_t;
    edge_list<int_t> elist;
    std::ifstream file(name);
    int_t source_vertex      = 0;
    int_t destination_vertex = 0;
    while (file >> source_vertex >> destination_vertex) {
        auto edge = std::make_pair(source_vertex, destination_vertex);
        elist.push_back(edge);
    }

    file.close();
    return elist;
}
} // namespace detail
} // namespace oneapi::dal::preview