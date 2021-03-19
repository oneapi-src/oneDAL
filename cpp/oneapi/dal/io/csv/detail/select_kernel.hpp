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

#include "oneapi/dal/io/csv/common.hpp"
#include "oneapi/dal/io/csv/detail/read_graph_kernel.hpp"

namespace oneapi::dal::csv::detail {

template <typename Policy, typename Descriptor, typename Graph, typename Allocator>
struct backend_base {
    virtual void operator()(const Policy &ctx,
                            const Descriptor &descriptor,
                            Graph &g,
                            const read_args<Graph, Allocator> &args) = 0;
    virtual ~backend_base() = default;
};

template <typename Policy, typename Descriptor, typename Graph, typename Allocator>
struct backend_default : public backend_base<Policy, Descriptor, Graph, Allocator> {
    static_assert(dal::detail::is_one_of_v<Policy, dal::detail::host_policy>,
                  "Host policy only is supported.");

    virtual void operator()(const Policy &ctx,
                            const Descriptor &descriptor,
                            Graph &g,
                            const read_args<Graph, Allocator> &args) {
        auto allocator = args.get_allocator();
        return read_graph_default_kernel(ctx, descriptor, allocator, g);
    }
};

template <typename Policy, typename Descriptor, typename Graph, typename Allocator>
dal::detail::shared<backend_base<Policy, Descriptor, Graph, Allocator>>
get_backend(const Descriptor &desc, Graph &data, const read_args<Graph, Allocator> &args) {
    return std::make_shared<backend_default<Policy, Descriptor, Graph, Allocator>>();
}

} // namespace oneapi::dal::csv::detail
