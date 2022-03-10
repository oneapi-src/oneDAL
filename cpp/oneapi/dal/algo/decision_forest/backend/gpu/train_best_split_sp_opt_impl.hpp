/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/algo/decision_forest/train_types.hpp"

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_misc_structs.hpp"
#include "oneapi/dal/algo/decision_forest/backend/gpu/train_impurity_data.hpp"

#include "oneapi/dal/algo/decision_forest/backend/gpu/train_node_helpers.hpp"

namespace oneapi::dal::decision_forest::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = bk::primitives;

template <typename Float,
          typename Bin = std::uint32_t,
          typename Index = std::int32_t,
          typename Task = task::by_default>
class train_best_split_sp_opt_impl {
    using result_t = train_result<Task>;
    using impl_const_t = impl_const<Index, Task>;
    using descriptor_t = detail::descriptor_base<Task>;
    using context_t = train_context<Float, Index, Task>;
    using imp_data_t = impurity_data<Float, Index, Task>;
    using msg = de::error_messages;
    using hist_type_t = typename task_types<Float, Index, Task>::hist_type_t;
    using node_list_t = node_list<Index>;
    //using node_group_list_t = node_group_list<Index>;
    using node_group_view_t = node_group_view<Index>;

public:
    train_best_split_sp_opt_impl() = default;
    ~train_best_split_sp_opt_impl() = default;

    static sycl::event compute_best_split_single_pass_large(
        sycl::queue& queue,
        const context_t& ctx,
        const pr::ndarray<Bin, 2>& data,
        const pr::ndview<Float, 1>& response,
        const pr::ndarray<Index, 1>& tree_order,
        const pr::ndarray<Index, 1>& selected_ftr_list,
        const pr::ndarray<Index, 1>& bin_offset_list,
        const imp_data_t& imp_data_list,
        const pr::ndarray<Index, 1>& node_ind_list,
        Index node_ind_ofs,
        pr::ndarray<Index, 1>& node_list,
        imp_data_t& left_child_imp_data_list,
        pr::ndarray<Float, 1>& node_imp_dec_list,
        bool update_imp_dec_required,
        Index node_count,
        const bk::event_vector& deps = {});
    /*
    static sycl::event compute_best_split_single_pass_large(
        sycl::queue& queue,
        const context_t& ctx,
        const pr::ndarray<Bin, 2>& data,
        const pr::ndview<Float, 1>& response,
        const pr::ndarray<Index, 1>& tree_order,
        const pr::ndarray<Index, 1>& selected_ftr_list,
        const pr::ndarray<Index, 1>& bin_offset_list,
        const imp_data_t& imp_data_list,
        const node_group_view_t& node_group,
        node_list_t& level_node_list,
        imp_data_t& left_child_imp_data_list,
        pr::ndarray<Float, 1>& node_imp_dec_list,
        bool update_imp_dec_required,
        const bk::event_vector& deps = {});
    */
    static sycl::event compute_best_split_single_pass_small(
        sycl::queue& queue,
        const context_t& ctx,
        const pr::ndarray<Bin, 2>& data,
        const pr::ndview<Float, 1>& response,
        const pr::ndarray<Index, 1>& tree_order,
        const pr::ndarray<Index, 1>& selected_ftr_list,
        const pr::ndarray<Index, 1>& bin_offset_list,
        const imp_data_t& imp_data_list,
        const node_group_view_t& node_group,
        node_list_t& level_node_list,
        imp_data_t& left_child_imp_data_list,
        pr::ndarray<Float, 1>& node_imp_dec_list,
        bool update_imp_dec_required,
        const bk::event_vector& deps = {});

    static std::int64_t define_local_size_for_small_single_pass(const sycl::queue& queue,
                                                                std::int64_t selected_ftr_count);
};

#endif

} // namespace oneapi::dal::decision_forest::backend
