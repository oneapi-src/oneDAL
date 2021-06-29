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

#include "oneapi/dal/algo/kmeans/train_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/detail/distributed.hpp"

namespace oneapi::dal::kmeans::detail {
namespace v1 {

template <typename Policy, typename Float, typename Method, typename Task, typename... Options>
struct train_ops_dispatcher {
    train_result<Task> operator()(const Policy&,
                                  const descriptor_base<Task>&,
                                  const train_input<Task>&) const;
};

template <typename Descriptor>
struct train_ops {
    using float_t = typename Descriptor::float_t;
    using method_t = typename Descriptor::method_t;
    using task_t = typename Descriptor::task_t;
    using input_t = train_input<task_t>;
    using result_t = train_result<task_t>;

    template <typename Policy>
    void check_preconditions(const Policy& policy,
                             const Descriptor& desc,
                             const input_t& input) const {
        using msg = dal::detail::error_messages;

        if (!(input.get_data().has_data())) {
            throw domain_error(msg::input_data_is_empty());
        }
        if (input.get_data().get_row_count() > dal::detail::limits<std::int32_t>::max()) {
            throw domain_error(dal::detail::error_messages::row_count_gt_max_int32());
        }
        // Need to count all the rows distributed across ranks
        // if (desc.get_cluster_count() > input.get_data().get_row_count()) {
        //     throw invalid_argument(msg::cluster_count_exceeds_data_row_count());
        // }
        if (input.get_initial_centroids().has_data()) {
            if (input.get_initial_centroids().get_row_count() != desc.get_cluster_count()) {
                throw invalid_argument(msg::input_initial_centroids_rc_neq_desc_cluster_count());
            }
            if (input.get_initial_centroids().get_column_count() !=
                input.get_data().get_column_count()) {
                throw invalid_argument(msg::input_initial_centroids_cc_neq_input_data_cc());
            }
        }
    }

    template <typename Policy>
    void check_postconditions(const Policy& policy,
                              const Descriptor& desc,
                              const input_t& input,
                              const result_t& result) const {
        ONEDAL_ASSERT(result.get_labels().has_data());
        ONEDAL_ASSERT(result.get_labels().get_column_count() == 1);
        ONEDAL_ASSERT(result.get_iteration_count() <= desc.get_max_iteration_count());
        ONEDAL_ASSERT(result.get_labels().get_row_count() == input.get_data().get_row_count());

        // K-Means model is initialized only on root rank
        if (dal::detail::is_root_rank(policy)) {
            ONEDAL_ASSERT(result.get_model().get_centroids().has_data());
            ONEDAL_ASSERT(result.get_model().get_centroids().get_row_count() ==
                          desc.get_cluster_count());
            ONEDAL_ASSERT(result.get_model().get_centroids().get_column_count() ==
                          input.get_data().get_column_count());
        }
    }

    template <typename Policy>
    auto operator()(const Policy& policy, const Descriptor& desc, const input_t& input) const {
        check_preconditions(policy, desc, input);
        const auto result =
            train_ops_dispatcher<Policy, float_t, method_t, task_t>()(policy, desc, input);
        check_postconditions(policy, desc, input, result);
        return result;
    }
};

} // namespace v1

using v1::train_ops;

} // namespace oneapi::dal::kmeans::detail
