/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_node_helpers.hpp"
#include "oneapi/dal/detail/profiler.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_misc_structs.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_helpers.hpp"

#ifdef ONEDAL_DATA_PARALLEL
namespace oneapi::dal::decision_forest::backend {

using alloc = sycl::usm::alloc;

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

template <typename Index>
sycl::event node_group_list<Index>::filter_internal(const node_list_t& node_list,
                                                    const bk::event_vector& deps) {
    ONEDAL_PROFILER_TASK(split_node_list_on_groups_by_size, queue_);

    ONEDAL_ASSERT(node_list.state_is_valid());
    ONEDAL_ASSERT(this->state_is_valid());

    const Index node_prop_count = node_t::get_prop_count(); // num of split attributes for node

    const Index* node_list_ptr = node_list.get_list().get_data();
    const Index node_count = node_list.get_count();

    auto node_groups = this->get_list();
    constexpr Index group_count = node_group_list_t::get_count();
    constexpr Index group_prop_count = node_group_t::get_prop_count();

    const Index* node_groups_bound_ptr = this->get_bound_list().get_data();
    Index* node_groups_ptr = node_groups.get_mutable_data();
    Index* node_indices_ptr = this->get_node_indices_list().get_mutable_data();

    Index max_sbg_size = bk::device_max_sg_size(queue_);
    Index local_size = std::max(Index(max_sbg_size * group_count), min_local_size_);
    ONEDAL_ASSERT(bk::device_max_wg_size(queue_) >= local_size);

    const sycl::nd_range<1> nd_range = bk::make_multiple_nd_range_1d(local_size, local_size);

    auto event = queue_.submit([&](sycl::handler& cgh) {
        cgh.depends_on(deps);
        local_accessor_rw_t<Index> local_buf(local_size, cgh);
        cgh.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
            auto sbg = item.get_sub_group();
            const Index sub_group_id = sbg.get_group_id();

            const Index local_id = sbg.get_local_id();
            const Index local_size = sbg.get_local_range()[0];
#if __SYCL_COMPILER_VERSION >= 20230828
            Index* local_buf_ptr =
                local_buf.template get_multi_ptr<sycl::access::decorated::yes>().get_raw();
#else
            Index* local_buf_ptr = local_buf.get_pointer().get();
#endif
            Index bucket_count = 0;
            Index max_row_count = 1;

            Index bucket_up_bound =
                sub_group_id < group_count ? node_groups_bound_ptr[sub_group_id] : 0;
            Index bucket_low_bound =
                sub_group_id < group_count ? node_groups_bound_ptr[sub_group_id + 1] : 0;

            for (Index i = local_id; i < node_count && sub_group_id < group_count;
                 i += local_size) {
                Index row_count = node_list_ptr[i * node_prop_count + node_t::ind_grc()];
                Index into_bucket =
                    (Index)(row_count <= bucket_up_bound && row_count > bucket_low_bound);

                bucket_count += sycl::reduce_over_group(sbg, into_bucket, plus<Index>());
                max_row_count = sycl::max(max_row_count, into_bucket ? row_count : 0);
                max_row_count = sycl::reduce_over_group(sbg, max_row_count, maximum<Index>());
            }

            local_buf_ptr[sub_group_id] = bucket_count;

            item.barrier(sycl::access::fence_space::local_space);

            Index prefix_sum = 0;
            for (Index i = 0; i < sub_group_id; ++i) {
                prefix_sum += local_buf_ptr[i];
            }

            if (0 == local_id && sub_group_id < group_count) {
                node_groups_ptr[sub_group_id * group_prop_count + 0] = prefix_sum;
                node_groups_ptr[sub_group_id * group_prop_count + 1] = max_row_count;

                node_groups_ptr[group_count * group_prop_count + 0] = node_count;
                node_groups_ptr[group_count * group_prop_count + 1] = 0;
            }

            //split nodes on groups
            for (Index i = local_id; i < node_count && bucket_count > 0; i += local_size) {
                Index row_count = node_list_ptr[i * node_prop_count + node_t::ind_grc()];
                Index into_bucket =
                    (Index)(row_count <= bucket_up_bound && row_count > bucket_low_bound);

                Index pos_new =
                    prefix_sum + sycl::exclusive_scan_over_group(sbg, into_bucket, plus<Index>());

                if (into_bucket) {
                    node_indices_ptr[pos_new] = i;
                }
                prefix_sum += sycl::reduce_over_group(sbg, into_bucket, plus<Index>());
            }
        });
    });

    return event;
}

#define INSTANTIATE(I) template class node_group_list<I>;

INSTANTIATE(std::int32_t);
} // namespace oneapi::dal::decision_forest::backend

#endif //#ifdef ONEDAL_DATA_PARALLEL
