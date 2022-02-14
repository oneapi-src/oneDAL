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

#pragma once

#include "oneapi/dal/algo/kmeans_init/compute_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::kmeans_init::detail {
namespace v1 {

template <typename Context, typename Float, typename Method, typename Task, typename... Options>
struct compute_ops_dispatcher {
    compute_result<Task> operator()(const Context&,
                                    const descriptor_base<Task>&,
                                    const compute_input<Task>&) const;
};

template <typename Descriptor>
struct compute_ops {
    using float_t = typename Descriptor::float_t;
    using task_t = typename Descriptor::task_t;
    using method_t = typename Descriptor::method_t;
    using input_t = compute_input<task_t>;
    using result_t = compute_result<task_t>;
    using descriptor_base_t = descriptor_base<task_t>;

    void check_preconditions(const Descriptor& params, const input_t& input) const {
        using msg = dal::detail::error_messages;

        if (!input.get_data().has_data()) {
            throw domain_error(msg::input_data_is_empty());
        }

        const auto& data_table = input.get_data();
        const auto sample_count = data_table.get_row_count();
        const auto cluster_count = params.get_cluster_count();

        if (0 >= cluster_count) {
            throw domain_error(msg::cluster_count_leq_zero());
        }

        if (cluster_count >= sample_count) {
            throw domain_error(msg::cluster_count_exceeds_data_row_count());
        }
    }

    void check_postconditions(const Descriptor& params,
                              const input_t& input,
                              const result_t& result) const {
        ONEDAL_ASSERT(result.get_centroids().has_data());
        ONEDAL_ASSERT(result.get_centroids().get_row_count() == params.get_cluster_count());
        ONEDAL_ASSERT(result.get_centroids().get_column_count() ==
                      input.get_data().get_column_count());
    }

    template <typename Context>
    auto operator()(const Context& ctx, const Descriptor& desc, const input_t& input) const {
        check_preconditions(desc, input);
        const auto result =
            compute_ops_dispatcher<Context, float_t, method_t, task_t>()(ctx, desc, input);
        check_postconditions(desc, input, result);
        return result;
    }
};

} // namespace v1

using v1::compute_ops;

} // namespace oneapi::dal::kmeans_init::detail
