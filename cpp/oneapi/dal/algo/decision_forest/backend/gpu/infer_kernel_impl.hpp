/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#include "oneapi/dal/algo/decision_forest/infer_types.hpp"
#include "oneapi/dal/detail/error_messages.hpp"

#include "oneapi/dal/algo/decision_forest/backend/gpu/infer_model_manager.hpp"

namespace oneapi::dal::decision_forest::backend {

#ifdef ONEDAL_DATA_PARALLEL

namespace bk = dal::backend;

template <typename Float, typename Index = std::int32_t, typename Task = task::by_default>
class infer_kernel_impl {
    using result_t = infer_result<Task>;
    using impl_const_t = infer_impl_const<Index, Task>;
    using descriptor_t = detail::descriptor_base<Task>;
    using model_manager_t = infer_model_manager<Float, Index, Task>;
    using infer_context_t = infer_context<Float, Index, Task>;
    using model_t = model<Task>;
    using msg = dal::detail::error_messages;
    using comm_t = bk::communicator<spmd::device_memory_access::usm>;

public:
    infer_kernel_impl(const bk::context_gpu& ctx)
            : queue_(ctx.get_queue()),
              comm_(ctx.get_communicator()) {}
    ~infer_kernel_impl() = default;

    result_t operator()(const descriptor_t& desc, const model_t& trained_model, const table& data);

private:
    void validate_input(const descriptor_t& desc,
                        const model_t& trained_model,
                        const table& data) const;

    void init_params(infer_context_t& ctx,
                     const descriptor_t& desc,
                     const model_t& trained_model,
                     const table& data) const;

    std::tuple<dal::backend::primitives::ndarray<Float, 1>, sycl::event>
    predict_by_tree_group_weighted(const infer_context_t& ctx,
                                   const dal::backend::primitives::ndview<Float, 2>& data,
                                   const model_manager_t& mng,
                                   const dal::backend::event_vector& deps = {});

    std::tuple<dal::backend::primitives::ndarray<Float, 1>, sycl::event> predict_by_tree_group(
        const infer_context_t& ctx,
        const dal::backend::primitives::ndview<Float, 2>& data,
        const model_manager_t& mng,
        const dal::backend::event_vector& deps = {});

    std::tuple<dal::backend::primitives::ndarray<Float, 1>, sycl::event> reduce_tree_group_response(
        const infer_context_t& ctx,
        const dal::backend::primitives::ndview<Float, 1>& obs_response_list,
        const dal::backend::event_vector& deps = {});

    std::tuple<dal::backend::primitives::ndarray<Float, 1>, sycl::event> determine_winner(
        const infer_context_t& ctx,
        const dal::backend::primitives::ndview<Float, 1>& response_list,
        const dal::backend::event_vector& deps = {});

    sycl::queue queue_;
    comm_t comm_;
};

#endif

} // namespace oneapi::dal::decision_forest::backend
