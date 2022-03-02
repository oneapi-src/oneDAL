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

#pragma once

#include "oneapi/dal/algo/pca/backend/gpu/train_kernel.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/util/common.hpp"
#include "oneapi/dal/detail/policy.hpp"
#include "oneapi/dal/backend/communicator.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::pca::backend {

namespace bk = dal::backend;

template <typename Float>
class train_kernel_cov_impl {
    using comm_t = bk::communicator<spmd::device_memory_access::usm>;
    using input_t = train_input<task::dim_reduction>;
    using result_t = train_result<task::dim_reduction>;
    using descriptor_t = detail::descriptor_base<task::dim_reduction>;

public:
    train_kernel_cov_impl(const bk::context_gpu& ctx)
            : q_(ctx.get_queue()),
              comm_(ctx.get_communicator()) {}
    result_t operator()(const descriptor_t& desc, const input_t& input);

private:
    sycl::queue q_;
    comm_t comm_;
};

} // namespace oneapi::dal::pca::backend
#endif // ONEDAL_DATA_PARALLEL
