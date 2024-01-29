/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "oneapi/dal/algo/covariance/backend/gpu/partial_compute_kernel.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/util/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/backend/communicator.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::covariance::backend {

namespace bk = dal::backend;

template <typename Float>
class partial_compute_kernel_dense_impl {
    using task_t = task::compute;
    using comm_t = bk::communicator<spmd::device_memory_access::usm>;
    using input_t = partial_compute_input<task_t>;
    using result_t = partial_compute_result<task_t>;
    using descriptor_t = detail::descriptor_base<task_t>;
    using parameters_t = detail::compute_parameters<task_t>;

public:
    partial_compute_kernel_dense_impl(const bk::context_gpu& ctx)
            : q(ctx.get_queue()),
              comm_(ctx.get_communicator()) {}
    result_t operator()(const descriptor_t& desc, const input_t& input);

private:
    sycl::queue q;
    comm_t comm_;
};

} // namespace oneapi::dal::covariance::backend

#endif // ONEDAL_DATA_PARALLEL
