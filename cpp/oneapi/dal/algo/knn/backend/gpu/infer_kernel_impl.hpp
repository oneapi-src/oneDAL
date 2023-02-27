/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#pragma once

namespace oneapi::dal::knn::backend {

using idx_t = std::int32_t;

template <typename Task>
using descriptor_t = detail::descriptor_base<Task>;

using voting_t = ::oneapi::dal::knn::voting_mode;

namespace de = ::oneapi::dal::detail;
namespace bk = ::oneapi::dal::backend;
namespace pr = ::oneapi::dal::backend::primitives;
namespace spmd = oneapi::dal::preview::spmd;

using comm_t = bk::communicator<spmd::device_memory_access::usm>;

template <typename Task, typename Float>
struct task_to_response_map {
    using type = int;
};

template <typename Float>
struct task_to_response_map<task::regression, Float> {
    using type = float;
};

template <typename Float>
struct task_to_response_map<task::classification, Float> {
    using type = std::int32_t;
};

template <typename Task, typename Float>
using response_t = typename task_to_response_map<Task, Float>::type;

} // namespace oneapi::dal::knn::backend
