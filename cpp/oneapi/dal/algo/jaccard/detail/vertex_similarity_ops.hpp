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

#include "oneapi/dal/algo/jaccard/common.hpp"
#include "oneapi/dal/algo/jaccard/vertex_similarity_types.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

namespace oneapi::dal::preview {
namespace jaccard {
namespace detail {

template <typename Policy, typename Float, class Method, typename Graph>
struct ONEDAL_EXPORT vertex_similarity_ops_dispatcher {
    vertex_similarity_result operator()(const Policy &policy,
                                        const descriptor_base &descriptor,
                                        vertex_similarity_input<Graph> &input) const;
};

template <typename Descriptor, typename Graph>
struct vertex_similarity_ops {
    using float_t = typename Descriptor::float_t;
    using method_t = typename Descriptor::method_t;
    using input_t = vertex_similarity_input<Graph>;
    using result_t = vertex_similarity_result;
    using descriptor_base_t = descriptor_base;

    void check_preconditions(const Descriptor &param, vertex_similarity_input<Graph> &input) const {
        using msg = dal::detail::error_messages;

        const auto row_begin = param.get_row_range_begin();
        const auto row_end = param.get_row_range_end();
        const auto column_begin = param.get_column_range_begin();
        const auto column_end = param.get_column_range_end();
        if (row_begin < 0 || column_begin < 0) {
            throw invalid_argument(msg::negative_interval());
        }
        if (row_begin > row_end) {
            throw invalid_argument(msg::row_begin_gt_row_end());
        }
        if (column_begin > column_end) {
            throw invalid_argument(msg::column_begin_gt_column_end());
        }
        const auto vertex_count =
            (dal::preview::detail::get_impl(input.get_graph())->_vertex_count);
        // Safe conversion as ranges were checked
        if (static_cast<std::size_t>(row_end) > vertex_count ||
            static_cast<std::size_t>(column_end) > vertex_count) {
            throw out_of_range(msg::interval_gt_vertex_count());
        }
        if (row_end >= dal::detail::limits<std::int32_t>::max() ||
            column_end >= dal::detail::limits<std::int32_t>::max()) {
            throw invalid_argument(msg::range_idx_gt_max_int32());
        }
    }

    template <typename Policy>
    auto operator()(const Policy &policy,
                    const Descriptor &desc,
                    vertex_similarity_input<Graph> &input) const {
        check_preconditions(desc, input);
        return vertex_similarity_ops_dispatcher<Policy, float_t, method_t, Graph>()(policy,
                                                                                    desc,
                                                                                    input);
    }
};

} // namespace detail
} // namespace jaccard
} // namespace oneapi::dal::preview
