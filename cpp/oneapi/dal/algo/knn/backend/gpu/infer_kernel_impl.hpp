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

#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/algo/knn/backend/model_conversion.hpp"
#include "oneapi/dal/algo/knn/backend/gpu/infer_kernel.hpp"
#include "oneapi/dal/algo/knn/backend/distance_impl.hpp"
#include "oneapi/dal/algo/knn/backend/model_impl.hpp"

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/regression.hpp"
#include "oneapi/dal/backend/primitives/search.hpp"
#include "oneapi/dal/backend/primitives/selection.hpp"
#include "oneapi/dal/backend/primitives/voting.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/backend/communicator.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/detail/common.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::knn::backend {

namespace bk = ::oneapi::dal::backend;

template <typename Float, typename Method = method::brute_force, typename Task>
class infer_kernel_knn_bf_impl {
    using comm_t = bk::communicator<spmd::device_memory_access::usm>;
    using model_t = model<Task>;
    using result_t = infer_result<Task>;
    using descriptor_t = detail::descriptor_base<Task>;

public:
    infer_kernel_knn_bf_impl(const bk::context_gpu& ctx)
            : q_(ctx.get_queue()),
              comm_(ctx.get_communicator()) {}
    result_t operator()(const descriptor_t& desc, const table& infer, const model_t& m);

private:
    sycl::queue q_;
    comm_t comm_;
};

} // namespace oneapi::dal::knn::backend
#endif // ONEDAL_DATA_PARALLEL