/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#pragma once

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/graph/common.hpp"
#include "oneapi/dal/io/csv/read_types.hpp"
// #include "oneapi/dal/backend/dispatcher.hpp"
#include "oneapi/dal/io/csv/detail/select_kernel.hpp"
#include <iostream>
// #include "oneapi/dal/io/csv/backend/cpu/read_kernel.hpp"

namespace oneapi::dal::csv::detail {
namespace v1 {

// template <typename Policy, typename Descriptor, typename Graph>
// struct backend_base {
//     // using float_t = typename Descriptor::float_t;
//     // using task_t = typename Descriptor::task_t;
//     // using method_t = typename Descriptor::method_t;
//     // using allocator_t = typename Descriptor::allocator_t;

//     virtual void operator()(const Policy &ctx, const Descriptor &descriptor, const Graph &g) = 0;
//     virtual ~backend_base() = default;
// };

// template <typename Policy, typename Descriptor, typename Graph>
// struct backend_default : public backend_base<Policy, Descriptor, Graph> {
//     static_assert(dal::detail::is_one_of_v<Policy, dal::detail::host_policy>,
//                   "Host policy only is supported.");

//     // using task_t = typename Descriptor::task_t;
//     // using allocator_t = typename Descriptor::allocator_t;

//     virtual void operator()(const Policy &ctx, const Descriptor &descriptor, const Graph &data) {
//         std::allocator<int> my_allocator;
//         return read_graph_default_kernel(ctx, descriptor, my_allocator, data);
//     }
// };

// template <typename Policy, typename Descriptor, typename Graph>
// dal::detail::shared<backend_base<Policy, Descriptor, Graph>> get_backend(const Descriptor &desc,
//                                                                          const Graph &g) {
//     return std::make_shared<backend_default<Policy, Descriptor, Graph>>();
// }

template <typename Object, typename Policy, typename... Options>
struct read_ops_dispatcher;

template <typename Graph> // Object = Graph
struct read_ops_dispatcher<Graph, dal::detail::host_policy> {
    Graph operator()(const dal::detail::host_policy& policy,
                     const data_source_base& ds,
                     const read_args<Graph>& args) const {
        Graph g;
        // const auto& csr_topology = dal::preview::detail::csr_topology_builder<Graph>()(g);
        // static auto impl = get_backend<Policy, Descriptor>(csr_topology);
        std::cout << "Graph" << std::endl;

        //         read_graph_default_kernel(const dal::detail::host_policy& ctx,
        //                                       const detail::data_source_base& ds,
        //                                       const Allocator& alloc,
        //                                       const Graph& g)
        std::allocator<int> my_allocator;
        static auto impl = get_backend<dal::detail::host_policy, data_source_base, Graph>(ds, g);
        (*impl)(policy, ds, g);
        return g; //(*impl)(policy, descriptor, csr_topology);
    }
};

template <>
struct read_ops_dispatcher<table, dal::detail::host_policy> {
    table operator()(const dal::detail::host_policy& policy,
                     const data_source_base& ds,
                     const read_args<table>& args) const;
};

template <typename Object, typename DataSource>
struct read_ops;

template <typename Object>
struct read_ops<Object, data_source> {
    // static_assert(std::is_same_v<Object, table> || std::is_same_v<Object, preview::graph_base>,
    //               "CSV data source is defined only for table and graphs");

    using args_t = read_args<Object>;
    using result_t = Object;

    void check_preconditions(const data_source_base& ds, const args_t& args) const {}

    void check_postconditions(const data_source_base& ds,
                              const args_t& args,
                              const result_t& result) const {}

    template <typename Policy>
    auto operator()(const Policy& ctx, const data_source_base& ds, const args_t& args) const {
        check_preconditions(ds, args);
        const auto result = read_ops_dispatcher<Object, Policy>()(ctx, ds, args);
        check_postconditions(ds, args, result);
        return result;
    }
};

} // namespace v1

using v1::read_ops;

} // namespace oneapi::dal::csv::detail
